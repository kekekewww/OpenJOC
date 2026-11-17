use super::*;
use crate::{DecodedAccessUnitPcm, DecodedJocAccessUnitPcm};

const FIXTURE: &[u8] = include_bytes!("../../openjoc-wasm/testdata/joc.lifecycle.ec3");

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
