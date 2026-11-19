use super::{
    DecodedAccessUnitPcm, DecodedPayloadFrame, OpenJocError, OpenJocPcmFrame,
    OpenJocReconstructionFrame, VecDeque,
};
use crate::reconstruction_positions::PositionCapture;
use openjoc_scene::{BaseFullBandCoordinate, BridgeControlFrame};

/// Only constructed for an explicitly requested reconstruction output.
#[derive(Debug)]
pub(super) struct ReconstructionCapture {
    delay_samples: usize,
    labels: Vec<String>,
    pending: VecDeque<f32>,
    ready: VecDeque<OpenJocReconstructionFrame>,
    positions: PositionCapture,
}

impl ReconstructionCapture {
    pub(super) fn new(delay_samples: usize) -> Self {
        Self {
            delay_samples,
            labels: Vec::new(),
            pending: VecDeque::new(),
            ready: VecDeque::new(),
            positions: PositionCapture::default(),
        }
    }

    pub(super) fn reset(&mut self) {
        self.labels.clear();
        self.pending.clear();
        self.ready.clear();
        self.positions = PositionCapture::default();
    }

    pub(super) fn push(
        &mut self,
        frame: &DecodedPayloadFrame,
        base: &DecodedAccessUnitPcm,
        base_coordinates: &[BaseFullBandCoordinate],
        control: &BridgeControlFrame,
        dimensions: usize,
    ) -> Result<(), OpenJocError> {
        let rows = &frame.decoded.reconstruction_basis.rows;
        let sample_count = rows.first().map_or(0, Vec::len);
        let admitted = frame.admitted_decoded_joc_binding().is_some();
        let mut labels = (0..rows.len())
            .map(|row| {
                if admitted {
                    format!("Dynamic object {} (JOC row {})", row + 1, row + 1)
                } else {
                    format!("Reconstruction row {}", row + 1)
                }
            })
            .collect::<Vec<_>>();
        if base.lfe.is_some() {
            labels.push("Base LFE".to_owned());
        }
        if self.labels.is_empty() {
            self.pending.resize(self.delay_samples * labels.len(), 0.0);
            self.labels = labels;
        } else if self.labels != labels {
            return Err(OpenJocError::Render(
                "reconstruction channel identity changed within a stream".to_owned(),
            ));
        }
        self.positions.capture(
            frame,
            base_coordinates,
            control,
            dimensions,
            self.delay_samples,
            base.lfe.is_some(),
        )?;
        for sample in 0..sample_count {
            for row in rows {
                self.pending.push_back(row[sample] as f32);
            }
            if let Some(lfe) = &base.lfe {
                self.pending.push_back(lfe[sample] as f32);
            }
        }
        let interleaved_f32 = self
            .pending
            .drain(..sample_count * self.labels.len())
            .collect();
        self.ready.push_back(OpenJocReconstructionFrame {
            sample_rate: frame.sample_rate,
            sample_count,
            pts_samples: None,
            channel_labels: self.labels.clone(),
            row_keys: self.positions.take(sample_count, sample_count),
            interleaved_f32,
        });
        Ok(())
    }

    pub(super) fn receive(
        &mut self,
        pcm: &OpenJocPcmFrame,
    ) -> Result<OpenJocReconstructionFrame, OpenJocError> {
        let mut frame = if let Some(frame) = self.ready.pop_front() {
            frame
        } else {
            let count = pcm.sample_count * self.labels.len();
            let audio_samples = self
                .pending
                .len()
                .checked_div(self.labels.len())
                .unwrap_or(0)
                .min(pcm.sample_count);
            let mut interleaved_f32 = Vec::with_capacity(count);
            for _ in 0..count {
                interleaved_f32.push(self.pending.pop_front().unwrap_or(0.0));
            }
            OpenJocReconstructionFrame {
                sample_rate: pcm.sample_rate,
                sample_count: pcm.sample_count,
                pts_samples: None,
                channel_labels: self.labels.clone(),
                row_keys: self.positions.take(pcm.sample_count, audio_samples),
                interleaved_f32,
            }
        };
        if frame.sample_count != pcm.sample_count || frame.sample_rate != pcm.sample_rate {
            return Err(OpenJocError::Render(
                "reconstruction output range mismatch".to_owned(),
            ));
        }
        frame.pts_samples = pcm.pts_samples;
        Ok(frame)
    }
}
