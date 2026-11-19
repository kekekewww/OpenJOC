use openjoc_scene::{
    BaseFullBandCoordinate, BridgeControlAssembler, BridgeControlFrame, DecodedPayloadFrame,
    SpatialBindingRecord, SpatialBindingState, SpatialSourceClass,
};
use std::collections::VecDeque;

use crate::OpenJocError;

/// Renderer routing of a raw output row. A dynamic object may stand still.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReconstructionRowKind {
    DynamicObject,
    /// A fixed Intermediate Spatial Format route, with its renderer label.
    /// Full-band speaker beds use the separate Base planes.
    FixedIsf(String),
    Lfe,
}

/// Target in effect at a sample of raw reconstruction PCM.
#[derive(Clone, Debug, PartialEq)]
pub struct ReconstructionKey {
    pub sample_offset: usize,
    pub kind: ReconstructionRowKind,
    /// Room XYZ, in ETSI TS 103 420 V1.2.1 clause 4.2.1 left-handed coordinates.
    /// X runs from left wall (0) to right wall (1), Y from front (0) to back
    /// (1), Z from floor (-1) to ceiling (1). The nominal listener is at
    /// (0.5, 0.5, 0). These are the bridge's clipped Q15 targets, including
    /// its use of the boundary intersection for positions at infinity.
    /// Absent for fixed routes, LFE, inactive objects and silence padding.
    pub position: Option<[f64; 3]>,
    /// Normalized width, depth and height in renderer order, when available.
    pub size: Option<[f64; 3]>,
    /// Linear spatial gain target. It has not been applied to the raw PCM.
    /// Zero for inactive rows and silence padding, unity for Base LFE audio.
    pub gain: f64,
    /// Duration supplied to the speaker gain scheduler, in base-rate samples.
    /// None means there is no new speaker ramp. This includes a carried
    /// target, an update on an axis omitted by the layout, padding and LFE.
    /// The renderer projects the position first and ramps each speaker gain.
    /// There is no intermediate Cartesian position during a ramp.
    pub ramp_duration: Option<u16>,
}

impl ReconstructionKey {
    fn from_record(record: &SpatialBindingRecord, ramp_duration: Option<u16>) -> Self {
        let descriptor = &record.descriptor;
        let kind = if descriptor.source_class == SpatialSourceClass::FixedLayout {
            ReconstructionRowKind::FixedIsf(descriptor.identity.clone())
        } else {
            ReconstructionRowKind::DynamicObject
        };
        Self {
            sample_offset: 0,
            kind,
            position: record
                .active
                .then(|| descriptor.coordinates.as_slice().try_into().ok())
                .flatten(),
            size: record.active.then_some(descriptor.extent).flatten(),
            gain: if record.active { record.scalar } else { 0.0 },
            ramp_duration,
        }
    }

    fn silence(&self) -> Self {
        Self {
            sample_offset: 0,
            kind: self.kind.clone(),
            position: None,
            size: None,
            gain: 0.0,
            ramp_duration: None,
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct PositionCapture {
    // Reduced-dimensional layouts discard axes before panning. Use the same
    // assembler with all three axes for the layout-independent room targets.
    room_assembler: Option<BridgeControlAssembler>,
    binding: SpatialBindingState,
    cursor: u64,
    pending: VecDeque<(u64, Vec<ReconstructionKey>)>,
    current: Vec<ReconstructionKey>,
}

impl PositionCapture {
    pub(super) fn capture(
        &mut self,
        frame: &DecodedPayloadFrame,
        base: &[BaseFullBandCoordinate],
        speaker_control: &BridgeControlFrame,
        dimensions: usize,
        delay: usize,
        lfe: bool,
    ) -> Result<(), OpenJocError> {
        let room_control;
        let control = if dimensions == 3 {
            speaker_control
        } else {
            room_control = self
                .room_assembler
                .get_or_insert_with(|| BridgeControlAssembler::new(64, 3))
                .assemble_frame(frame, base, None)?;
            &room_control
        };
        let count = base.len() + frame.decoded.reconstruction_basis.rows.len();
        let start = self.cursor + delay as u64;
        let mut topology = control.initial_topology.as_ref();
        // A change on an omitted axis can update XYZ without starting a
        // speaker ramp. Conversely a reduced layout has its own quantization.
        let mut room_events = control.events.iter().peekable();
        let mut speaker_events = speaker_control.events.iter().peekable();
        while let Some(quantum) = [
            topology.map(|_| 0),
            room_events.peek().map(|event| event.quantum),
            speaker_events.peek().map(|event| event.quantum),
        ]
        .into_iter()
        .flatten()
        .min()
        {
            let updates = room_events
                .next_if(|event| event.quantum == quantum)
                .map(|event| event.updates.as_slice());
            let ramp = speaker_events
                .next_if(|event| event.quantum == quantum)
                .map(|event| event.ramp_duration)
                .or(topology.map(|_| 0));
            if topology.is_some() || updates.is_some() {
                self.binding
                    .apply(topology.take(), updates, count)
                    .map_err(|error| OpenJocError::Render(error.to_string()))?;
            }
            self.queue(start + quantum * 32, base.len(), lfe, ramp);
        }
        Ok(())
    }

    fn queue(&mut self, at: u64, base: usize, lfe: bool, ramp: Option<u16>) {
        // A speaker event cannot precede the first topology, but a missing
        // snapshot must not stop the decoder: there is nothing to report yet.
        let Some(snapshot) = self.binding.snapshot() else {
            return;
        };
        let records = &snapshot.records;
        let mut keys: Vec<_> = records[base..]
            .iter()
            .map(|record| ReconstructionKey::from_record(record, ramp))
            .collect();
        if lfe {
            keys.push(ReconstructionKey {
                sample_offset: 0,
                kind: ReconstructionRowKind::Lfe,
                position: None,
                size: None,
                gain: 1.0,
                ramp_duration: None,
            });
        }
        if self.current.is_empty() {
            self.current = keys.iter().map(ReconstructionKey::silence).collect();
        }
        self.pending.push_back((at, keys));
    }

    pub(super) fn take(
        &mut self,
        count: usize,
        audio_samples: usize,
    ) -> Vec<Vec<ReconstructionKey>> {
        let mut rows: Vec<_> = self.current.iter().map(|key| vec![key.clone()]).collect();
        let end = self.cursor + count as u64;
        while self.pending.front().is_some_and(|(at, _)| *at < end) {
            let (at, keys) = self.pending.pop_front().expect("checked front");
            for ((row, current), mut key) in rows.iter_mut().zip(&mut self.current).zip(keys) {
                if key.ramp_duration.is_none() && key == *current {
                    continue;
                }
                *current = key.clone();
                current.ramp_duration = None;
                key.sample_offset = (at - self.cursor) as usize;
                if key.sample_offset == 0 {
                    row.clear();
                }
                row.push(key);
            }
        }
        if audio_samples < count {
            for (row, current) in rows.iter_mut().zip(&mut self.current) {
                *current = current.silence();
                let mut key = current.clone();
                key.sample_offset = audio_samples;
                if audio_samples == 0 {
                    row.clear();
                }
                row.push(key);
            }
        }
        self.cursor = end;
        rows
    }
}
