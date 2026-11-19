use super::*;
use crate::ReconstructionRowKind;
use crate::{DecodedAccessUnitPcm, DecodedJocAccessUnitPcm};

#[allow(dead_code)]
mod synthetic {
    include!("../../openjoc-inspect/tests/support/mod.rs");

    // Same single-object syntax as active_object_oamd. Spell out the
    // not-active bit and the full six-bit X and Y fields for known positions.
    pub fn positioned_frame(
        sequence: u64,
        xy: [u64; 2],
        block_offset: u64,
        ramp_code: u64,
    ) -> Vec<u8> {
        let mut prefix = Vec::new();
        for (value, width) in [(0, 2), (0, 5), (1, 1), (0, 1), (0, 1), (1, 4)] {
            push(&mut prefix, value, width);
        }
        let mut content = Vec::new();
        for (value, width) in [
            (0, 1),
            (0, 2),
            (0, 3),
            (block_offset, 6),
            (ramp_code, 2),
            (1, 1),
            (0, 1), // active
            (2, 2),
            (20, 6),
            (1, 1), // -6 dB and default priority
            (xy[0], 6),
            (xy[1], 6),
            (1, 1),
            (6, 4),
            (0, 1), // XYZ and inside room
            (0, 3),
            (0, 1),
            (0, 2),
            (0, 1),
            (0, 1),
        ] {
            push(&mut content, value, width);
        }
        push_element(&mut prefix, 1, 10, &content);
        let mut joc = vec![false; 20]; // one row, five-channel downmix
        push(&mut joc, sequence, 10);
        push(&mut joc, 0, 1); // absent matrix
        five_channel_audio_frame(&joc_emdf(&pack(prefix), &pack(joc)))
    }
}

#[test]
fn position_targets_follow_the_speaker_bridge_on_the_pcm_timeline() {
    use openjoc_scene::{bridge_gain_scalar, bridge_quantize};

    for layout in ["7.1.4", "2.0"] {
        let config = OpenJocConfig {
            speaker_layout: layout.into(),
            ..Default::default()
        };
        let mut plain = OpenJocSession::new(config.clone()).unwrap();
        let mut probe = OpenJocSession::new(config.clone()).unwrap();
        probe.speaker.linked_gain_enabled = false;
        let mut probe_blocks = Vec::new();
        let mut paired = OpenJocSession::new_with_reconstruction(config).unwrap();
        let mut track = Vec::new();
        let mut samples = 0;
        for sequence in 0..5 {
            // A 512-sample route ramp begins at the final quantum of the third AU.
            let bytes = if sequence < 2 {
                synthetic::positioned_frame(sequence, [8, 12], 0, 0)
            } else {
                synthetic::positioned_frame(sequence, [54, 12], 47, 1)
            };
            let packet = OpenJocPacket {
                data: &bytes,
                pts_samples: None,
                discontinuity: false,
                preroll: false,
            };
            plain.push_packet(packet).unwrap();
            paired.push_packet(packet).unwrap();
            let pending = plain.speaker.pending_frames.back().unwrap();
            probe_blocks.extend(render_unit_row(&mut probe.speaker, pending));
            while let Some((pcm, frame)) = paired.receive_frame_with_reconstruction() {
                assert_eq!(plain.receive_frame().unwrap(), pcm);
                assert_eq!(frame.row_keys.len(), 1);
                let keys = &frame.row_keys[0];
                assert_eq!(keys[0].sample_offset, 0);
                assert!(
                    keys.windows(2)
                        .all(|w| w[0].sample_offset < w[1].sample_offset)
                );
                for key in keys {
                    assert_eq!(key.kind, ReconstructionRowKind::DynamicObject);
                    assert!(key.sample_offset < frame.sample_count);
                    if key.ramp_duration.is_some() {
                        track.push((samples + key.sample_offset, key.clone()));
                    }
                }
                samples += frame.sample_count;
            }
        }
        assert_eq!(track.len(), 2);
        let delay = FINAL_LINKED_GAIN_LATENCY_SAMPLES;
        assert_eq!(track[0].0, delay);
        assert_eq!(track[1].0, 2 * 1536 + 47 * 32 + delay);
        assert_eq!(track[1].1.ramp_duration, Some(512));
        for ((_, key), x) in track.iter().zip([8.0, 54.0]) {
            assert_eq!(
                key.position,
                Some([
                    bridge_quantize(x / 62.0, false),
                    bridge_quantize(12.0 / 62.0, false),
                    bridge_quantize(6.0 / 15.0, true),
                ])
            );
            assert_eq!(
                key.gain,
                bridge_gain_scalar(openjoc_oamd::Gain::Decibels(-6)).unwrap()
            );
            assert_eq!(key.size, Some([0.0; 3]));
        }
        // Check the independent, ordinary speaker renderer at its final sample.
        let state = plain.speaker.bridge.binding_state().snapshot().unwrap();
        let descriptor = &state.records.last().unwrap().descriptor;
        let point = track.last().unwrap().1.position.unwrap();
        assert_eq!(
            descriptor.coordinates,
            openjoc_scene::bridge_position(
                openjoc_oamd::Position3 {
                    x: point[0],
                    y: point[1],
                    z: point[2]
                },
                plain.speaker.layout.spatial().coordinate_dimension_count(),
            )
            .unwrap()
        );
        // The unit-input ordinary render measures the actual gain of each
        // route. Compare every sample across the AU boundary to the ramp
        // implied by projecting the two returned targets onto this layout.
        let spatial = plain.speaker.layout.spatial();
        let routes = track
            .iter()
            .map(|(_, key)| {
                let [x, y, z] = key.position.unwrap();
                let mut target = descriptor.clone();
                target.coordinates = openjoc_scene::bridge_position(
                    openjoc_oamd::Position3 { x, y, z },
                    spatial.coordinate_dimension_count(),
                )
                .unwrap();
                spatial
                    .project(&target)
                    .unwrap()
                    .into_iter()
                    .map(|v| v * key.gain)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_ne!(routes[0], routes[1]);
        let active = spatial
            .channels()
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.lfe);
        let duration = usize::from(track[1].1.ramp_duration.unwrap());
        for (route, (channel, _)) in active.enumerate() {
            let mut scheduler = openjoc_scene::GainScheduler::new();
            scheduler
                .set_target(routes[0][route], true, 0, 48_000)
                .unwrap();
            scheduler
                .set_target(routes[1][route], true, duration as u64, 48_000)
                .unwrap();
            for elapsed in 0..=duration {
                let sample = track[1].0 - delay + elapsed;
                let measured = probe_blocks[sample / 1536].channels[channel][sample % 1536];
                let expected = scheduler.next_sample();
                assert!(
                    (measured - expected).abs() < 1e-9,
                    "{layout} channel {channel} sample {sample}: {measured} != {expected}"
                );
            }
        }
    }
}

fn render_unit_row(
    speaker: &mut SpeakerRenderer,
    pending: &crate::PendingRenderFrame,
) -> Vec<RenderedBlock> {
    let mut frame = pending.frame.clone();
    frame.decoded.reconstruction_basis.rows[0].fill(1.0);
    let base = DecodedAccessUnitPcm {
        sample_rate: 48_000,
        samples: 1536,
        channel_locations: pending.channel_locations.clone(),
        channels: vec![vec![0.0; 1536]; pending.channel_locations.len()],
        lfe_location: None,
        lfe: None,
        downmix: pending.downmix,
        dialnorm: pending.dialnorm,
        dialnorm_segments: pending.dialnorm_segments.clone(),
    };
    speaker
        .render_frame_aligned(
            &frame,
            &DecodedJocAccessUnitPcm {
                compatibility_pcm: base.clone(),
                joc_input_pcm: base,
            },
        )
        .unwrap()
}

const FIXTURE: &[u8] = include_bytes!("../../openjoc-wasm/testdata/joc.lifecycle.ec3");

#[test]
fn one_axis_layout_retains_depth_updates_that_do_not_start_a_speaker_ramp() {
    use openjoc_scene::{SpatialLayout, SpatialLayoutNode};

    let mut preset = crate::SpeakerLayoutPreset::for_name("2.0").unwrap();
    preset.layout = SpatialLayout::new(
        preset.layout.channels().to_vec(),
        vec![vec![0.0, 1.0]],
        vec![
            SpatialLayoutNode {
                knot_indices: vec![0],
                vector: vec![1.0, 0.0],
            },
            SpatialLayoutNode {
                knot_indices: vec![1],
                vector: vec![0.0, 1.0],
            },
        ],
        Vec::new(),
    )
    .unwrap();
    let config =
        OpenJocConfig::default().with_speaker_layout(crate::SpeakerLayout::from_preset(preset));
    let mut session = OpenJocSession::new_with_reconstruction(config).unwrap();
    let mut states = Vec::new();
    let mut start = 0;
    for sequence in 0..3 {
        let y = if sequence == 0 { 12 } else { 50 };
        let bytes = synthetic::positioned_frame(sequence, [8, y], 1, 1);
        session
            .push_packet(OpenJocPacket {
                data: &bytes,
                pts_samples: None,
                discontinuity: false,
                preroll: false,
            })
            .unwrap();
        while let Some((_, frame)) = session.receive_frame_with_reconstruction() {
            for key in &frame.row_keys[0] {
                if let Some(position) = key.position.filter(|_| key.sample_offset > 0) {
                    states.push((start + key.sample_offset, position[1], key.ramp_duration));
                }
            }
            start += frame.sample_count;
        }
    }
    assert_eq!(
        states,
        [
            (
                64,
                openjoc_scene::bridge_quantize(12.0 / 62.0, false),
                Some(512)
            ),
            (
                1536 + 64,
                openjoc_scene::bridge_quantize(50.0 / 62.0, false),
                None
            ),
        ]
    );
}

#[test]
fn fixed_isf_routes_exclude_the_separate_speaker_bed() {
    use crate::reconstruction_positions::PositionCapture;
    use openjoc_oamd::{BedAssignment, ContentDescription, OamdElement};
    use openjoc_scene::{BaseFullBandCoordinate, BridgeControlAssembler};

    let mut session = OpenJocSession::new(OpenJocConfig::default()).unwrap();
    session.push_packet(input_packets(0)[0]).unwrap();
    let mut frame = session
        .speaker
        .pending_frames
        .front()
        .unwrap()
        .frame
        .clone();
    // Scene bridge mixed topology: a Base centre bed, four fixed ISF
    // routes and a dynamic row. The packet-level decoder currently admits
    // only the dynamic count as reconstruction rows.
    frame.oamd.prefix.object_count = 6;
    frame.oamd.prefix.content = ContentDescription::Mixed {
        bed_channel_distribute: Some(false),
        beds: vec![BedAssignment::Standard(1 << 8)],
        intermediate_spatial_format: Some(0),
        dynamic_objects: Some(1),
    };
    frame
        .decoded
        .reconstruction_basis
        .rows
        .resize(5, vec![0.0; 1536]);
    for element in &mut frame.oamd.elements {
        if let OamdElement::Objects(objects) = &mut element.element {
            objects.objects.resize(6, objects.objects[0].clone());
        }
    }
    let base = [BaseFullBandCoordinate::Centre];
    let control = BridgeControlAssembler::new(64, 3)
        .assemble_frame(&frame, &base, None)
        .unwrap();
    let mut capture = PositionCapture::default();
    capture
        .capture(&frame, &base, &control, 3, 0, false)
        .unwrap();
    let rows = capture.take(1536, 1536);
    assert_eq!(rows.len(), 5);
    for (keys, label) in rows[..4].iter().zip(["M1", "M2", "M3", "U1"]) {
        assert!(keys.iter().all(
            |key| key.kind == ReconstructionRowKind::FixedIsf(label.into())
                && key.position.is_none()
        ));
    }
    assert_eq!(rows[4][0].kind, ReconstructionRowKind::DynamicObject);
}

fn input_packets(start: usize) -> Vec<OpenJocPacket<'static>> {
    let frames = index_syncframes(FIXTURE).unwrap();
    let units = group_access_units(&frames).unwrap();
    units
        .iter()
        .enumerate()
        .skip(start)
        .map(|(index, unit)| {
            let first = &frames[unit.first_frame];
            let last = &frames[unit.first_frame + unit.frame_count - 1];
            OpenJocPacket {
                data: &FIXTURE[first.offset..last.offset + last.header.frame_size],
                pts_samples: Some(i64::try_from(index * 1536).unwrap()),
                discontinuity: false,
                preroll: false,
            }
        })
        .collect()
}

#[test]
fn reconstruction_impulses_align_with_render_through_timeline_and_gain() {
    let mut template_session = OpenJocSession::new(OpenJocConfig::default()).unwrap();
    template_session.push_packet(input_packets(0)[0]).unwrap();
    let pending = template_session.speaker.pending_frames.front().unwrap();
    let template = pending.frame.clone();
    let mut base = DecodedAccessUnitPcm {
        sample_rate: 48_000,
        samples: 1536,
        channel_locations: pending.channel_locations.clone(),
        channels: vec![vec![0.0; 1536]; pending.channel_locations.len()],
        lfe_location: pending.lfe_location,
        lfe: Some(vec![0.0; 1536]),
        downmix: pending.downmix,
        dialnorm: pending.dialnorm,
        dialnorm_segments: pending.dialnorm_segments.clone(),
    };
    // The source impulse enters the reconstruction QMF timeline later than
    // Base LFE, just as the causal reconstruction filterbank delivers it.
    // Measure the actual output positions instead of accepting a delay sum.
    let impulses = [0_usize, 31, 900, 1535, 1536, 2900];
    for layout in ["5.1", "7.1.4", "binaural"] {
        if layout == "binaural" && !cfg!(feature = "embedded-builtin-hrtf") {
            continue;
        }
        let mut session = OpenJocSession::new_with_reconstruction(OpenJocConfig {
            speaker_layout: if layout == "binaural" {
                "7.1.4"
            } else {
                layout
            }
            .to_owned(),
            render_mode: if layout == "binaural" {
                RenderMode::Binaural
            } else {
                RenderMode::Speaker
            },
            binaural: (layout == "binaural").then(|| {
                let mut config = crate::BinauralConfig::builtin_generic("7.1.4");
                config.lfe_policy = crate::BinauralLfePolicy::EqualPowerDualMono;
                config
            }),
            ..OpenJocConfig::default()
        })
        .unwrap();
        session.sample_rate = Some(48_000);
        session.segment_pts = Some(0);
        let mut renders = Vec::new();
        let mut objects = Vec::new();
        let mut render_channels = 0;
        let mut object_channels = 0;
        for index in 0..3 {
            let mut frame = template.clone();
            frame.frame_index = index as u64;
            frame.sample_range.start_sample = (index * 1536) as u64;
            frame.sample_range.end_sample = ((index + 1) * 1536) as u64;
            frame.decoded.state_reset = index == 0;
            for row in &mut frame.decoded.reconstruction_basis.rows {
                row.fill(0.0);
                for &impulse in &impulses {
                    let physical = impulse + QMF_LATENCY_SAMPLES;
                    if physical / 1536 == index && layout != "binaural" {
                        row[physical % 1536] = 0.125;
                    }
                }
            }
            let lfe = base.lfe.as_mut().unwrap();
            lfe.fill(0.0);
            for &impulse in &impulses {
                if impulse / 1536 == index {
                    lfe[impulse % 1536] = 0.125;
                }
            }
            let planes = DecodedJocAccessUnitPcm {
                compatibility_pcm: base.clone(),
                joc_input_pcm: base.clone(),
            };
            let blocks = session
                .speaker
                .render_frame_aligned(&frame, &planes)
                .unwrap();
            session.emit_rendered(blocks).unwrap();
            while let Some((render, reconstruction)) = session.receive_frame_with_reconstruction() {
                render_channels = render.channel_count;
                object_channels = reconstruction.channel_labels.len();
                renders.extend(render.interleaved_f32);
                objects.extend(reconstruction.interleaved_f32);
            }
        }
        let render_lfe = if layout == "binaural" {
            0
        } else {
            session
                .output_info()
                .channel_labels
                .iter()
                .position(|label| label == "LFE")
                .unwrap()
        };
        let positions = |pcm: &[f32], channels: usize, channel: usize| {
            pcm.chunks_exact(channels)
                .enumerate()
                .filter_map(|(index, samples)| (samples[channel] != 0.0).then_some(index))
                .collect::<Vec<_>>()
        };
        let actual = positions(&renders, render_channels, render_lfe);
        assert_eq!(actual.len(), impulses.len());
        for channel in 0..object_channels {
            if layout == "binaural" && channel + 1 < object_channels {
                assert_eq!(
                    positions(&objects, object_channels, channel),
                    [] as [usize; 0]
                );
                continue;
            }
            assert_eq!(
                positions(&objects, object_channels, channel),
                actual,
                "row/LFE {channel} impulse positions disagree with real {layout} render"
            );
            for &position in &actual {
                assert_eq!(objects[position * object_channels + channel], 0.125);
            }
        }
        println!(
            "{layout}: actual render LFE positions {actual:?}; all {object_channels} raw channels coincide: measured lag 0 samples (20.833 microsecond sample resolution)"
        );
    }
}

fn diagnostic_decode(start: usize, linked_gain: bool) -> Vec<f32> {
    let mut session = OpenJocSession::new(OpenJocConfig::default()).unwrap();
    session.speaker.linked_gain_enabled = linked_gain;
    let mut samples = Vec::new();
    for packet in input_packets(start) {
        session.push_packet(packet).unwrap();
        while let Some(frame) = session.receive_frame() {
            samples.extend(frame.interleaved_f32);
        }
    }
    session.drain().unwrap();
    while let Some(frame) = session.receive_frame() {
        samples.extend(frame.interleaved_f32);
    }
    samples
}

#[test]
#[ignore = "manual full-stream cold-start gain-state diagnostic"]
fn lifecycle_gain_state_diagnostic() {
    for enabled in [true, false] {
        let reference = diagnostic_decode(0, enabled);
        for start in [1, 16, 64] {
            let cold = diagnostic_decode(start, enabled);
            let reference = &reference[start * 1536 * 6..];
            let mut last = None;
            let mut tail_max = 0.0_f32;
            for (index, (&actual, &expected)) in cold.iter().zip(reference).enumerate() {
                let difference = (actual - expected).abs();
                if difference > 1e-6 {
                    last = Some(index / 6);
                }
                if index >= cold.len() / 2 {
                    tail_max = tail_max.max(difference);
                }
            }
            println!(
                "lifecycle linked_gain={enabled} cold_AU={start}: last_diff_gt_1e-6={last:?} tail_half_max={tail_max:.9}"
            );
        }
    }
}
