use openjoc_api::{
    BinauralConfig, OpenJocConfig, OpenJocPacket, OpenJocPcmFrame, OpenJocReconstructionFrame,
    OpenJocSession, OpenJocStatus, RenderMode,
};
use openjoc_eac3::{group_access_units, index_syncframes};

const FIXTURES: &[&[u8]] = &[
    include_bytes!("../../openjoc-wasm/testdata/joc.ec3"),
    include_bytes!("../../openjoc-wasm/testdata/joc.lifecycle.ec3"),
];

fn configurations() -> Vec<OpenJocConfig> {
    let mut configurations = vec![
        OpenJocConfig::default(),
        OpenJocConfig {
            speaker_layout: "7.1.4".to_owned(),
            ..OpenJocConfig::default()
        },
        OpenJocConfig {
            render_mode: RenderMode::Stereo,
            speaker_layout: "2.0".to_owned(),
            ..OpenJocConfig::default()
        },
    ];
    if cfg!(feature = "embedded-builtin-hrtf") {
        configurations.push(OpenJocConfig {
            render_mode: RenderMode::Binaural,
            speaker_layout: "7.1.4".to_owned(),
            binaural: Some(BinauralConfig::builtin_generic("7.1.4")),
            ..OpenJocConfig::default()
        });
    }
    configurations
}

fn packets(bytes: &[u8]) -> Vec<OpenJocPacket<'_>> {
    let frames = index_syncframes(bytes).unwrap();
    let units = group_access_units(&frames).unwrap();
    let mut pts_samples = -12_345;
    units
        .iter()
        .map(|unit| {
            let first = &frames[unit.first_frame];
            let last = &frames[unit.first_frame + unit.frame_count - 1];
            let packet = OpenJocPacket {
                data: &bytes[first.offset..last.offset + last.header.frame_size],
                pts_samples: Some(pts_samples),
                discontinuity: false,
                preroll: false,
            };
            pts_samples += i64::from(unit.samples);
            packet
        })
        .collect()
}

fn collect_pairs(
    session: &mut OpenJocSession,
    pairs: &mut Vec<(OpenJocPcmFrame, OpenJocReconstructionFrame)>,
) {
    while let Some(pair) = session.receive_frame_with_reconstruction() {
        pairs.push(pair);
    }
}

fn decode_pairs(
    session: &mut OpenJocSession,
    bytes: &[u8],
) -> Vec<(OpenJocPcmFrame, OpenJocReconstructionFrame)> {
    let mut pairs = Vec::new();
    for packet in packets(bytes) {
        assert_ne!(
            session.push_packet(packet).unwrap(),
            OpenJocStatus::OutputPending
        );
        collect_pairs(session, &mut pairs);
    }
    session.drain().unwrap();
    collect_pairs(session, &mut pairs);
    assert!(session.is_drained());
    pairs
}

#[test]
fn reconstruction_leaves_render_bit_identical_in_every_render_mode() {
    for (fixture_index, fixture) in FIXTURES.iter().enumerate() {
        for config in configurations() {
            let mut plain = OpenJocSession::new(config.clone()).unwrap();
            let mut paired = OpenJocSession::new_with_reconstruction(config).unwrap();
            let mut ordinary_frames = Vec::new();
            for packet in packets(fixture) {
                plain.push_packet(packet).unwrap();
                while let Some(frame) = plain.receive_frame() {
                    ordinary_frames.push(frame);
                }
            }
            plain.drain().unwrap();
            while let Some(frame) = plain.receive_frame() {
                ordinary_frames.push(frame);
            }
            let pairs = decode_pairs(&mut paired, fixture);
            assert_eq!(pairs.len(), ordinary_frames.len());
            assert_ne!(pairs.len(), 0);
            let labels = &pairs[0].1.channel_labels;
            let expected_labels = if fixture_index == 0 {
                vec!["Reconstruction row 1"]
            } else {
                vec!["Reconstruction row 1", "Base LFE"]
            };
            assert_eq!(labels, &expected_labels);
            let mut nonzero = false;
            for ((render, reconstruction), expected) in pairs.iter().zip(&ordinary_frames) {
                assert_eq!(render, expected);
                assert!(
                    render
                        .interleaved_f32
                        .iter()
                        .zip(&expected.interleaved_f32)
                        .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
                );
                assert_eq!(reconstruction.sample_rate, render.sample_rate);
                assert_eq!(reconstruction.sample_count, render.sample_count);
                assert_eq!(reconstruction.pts_samples, render.pts_samples);
                assert_eq!(&reconstruction.channel_labels, labels);
                assert_eq!(
                    reconstruction.interleaved_f32.len(),
                    render.sample_count * labels.len()
                );
                assert!(
                    reconstruction
                        .interleaved_f32
                        .iter()
                        .all(|value| value.is_finite())
                );
                nonzero |= reconstruction
                    .interleaved_f32
                    .iter()
                    .any(|value| *value != 0.0);
            }
            assert!(nonzero);
        }
    }
}

#[test]
fn reconstruction_reset_and_discontinuity_discard_old_state() {
    let fixture = &FIXTURES[1][..6 * 4096];
    for config in configurations() {
        let mut session = OpenJocSession::new_with_reconstruction(config).unwrap();
        let reference = decode_pairs(&mut session, fixture);
        session.reset();
        assert_eq!(decode_pairs(&mut session, fixture), reference);
        session.reset();
        let input = packets(fixture);
        session.push_packet(input[0]).unwrap();
        session.push_packet(input[1]).unwrap();
        while session.receive_frame_with_reconstruction().is_some() {}
        let mut discontinuity = input[0];
        discontinuity.discontinuity = true;
        assert_ne!(
            session.push_packet(discontinuity).unwrap(),
            OpenJocStatus::OutputPending
        );
        let mut actual = Vec::new();
        collect_pairs(&mut session, &mut actual);
        for packet in input.into_iter().skip(1) {
            session.push_packet(packet).unwrap();
            collect_pairs(&mut session, &mut actual);
        }
        session.drain().unwrap();
        collect_pairs(&mut session, &mut actual);
        assert_eq!(actual, reference);
    }
}

#[test]
fn ordinary_receive_discards_matching_reconstruction() {
    let fixture = &FIXTURES[1][..6 * 4096];
    let config = OpenJocConfig::default();
    let mut reference_session = OpenJocSession::new_with_reconstruction(config.clone()).unwrap();
    let reference = decode_pairs(&mut reference_session, fixture);
    let mut session = OpenJocSession::new_with_reconstruction(config).unwrap();
    let mut output_index = 0;
    let mut receive = |session: &mut OpenJocSession| loop {
        if output_index % 2 == 0 {
            let Some(render) = session.receive_frame() else {
                break;
            };
            assert_eq!(render, reference[output_index].0);
        } else {
            let Some(pair) = session.receive_frame_with_reconstruction() else {
                break;
            };
            assert_eq!(pair, reference[output_index]);
        }
        output_index += 1;
    };
    for packet in packets(fixture) {
        session.push_packet(packet).unwrap();
        receive(&mut session);
    }
    session.drain().unwrap();
    receive(&mut session);
    assert_eq!(output_index, reference.len());
    assert!(session.is_drained());
}

#[test]
fn disabled_reconstruction_receive_does_not_consume_render() {
    let fixture = FIXTURES[0];
    let mut session = OpenJocSession::new(OpenJocConfig::default()).unwrap();
    let mut count = 0;
    for packet in packets(fixture) {
        session.push_packet(packet).unwrap();
        assert!(session.receive_frame_with_reconstruction().is_none());
        while session.receive_frame().is_some() {
            count += 1;
        }
    }
    session.drain().unwrap();
    assert!(session.receive_frame_with_reconstruction().is_none());
    while session.receive_frame().is_some() {
        count += 1;
    }
    assert!(count > 0);
    assert!(session.is_drained());
}

#[test]
fn flush_discards_queued_pairs_and_retains_reconstruction_configuration() {
    let fixture = &FIXTURES[1][..6 * 4096];
    for config in configurations() {
        let mut reference_session =
            OpenJocSession::new_with_reconstruction(config.clone()).unwrap();
        let reference = decode_pairs(&mut reference_session, fixture);
        let mut session = OpenJocSession::new_with_reconstruction(config).unwrap();
        let descriptor = session.effective_config_descriptor();
        let input = packets(fixture);
        session.push_packet(input[0]).unwrap();
        assert_eq!(
            session.push_packet(input[1]).unwrap(),
            OpenJocStatus::FrameAvailable
        );
        session.flush();
        assert!(session.receive_frame().is_none());
        assert!(session.receive_frame_with_reconstruction().is_none());
        assert_eq!(session.effective_config_descriptor(), descriptor);
        assert_eq!(decode_pairs(&mut session, fixture), reference);
    }
}
