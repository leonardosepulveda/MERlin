import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from skimage import registration

from merlin.analysis import globalalign
from merlin.analysis import warp
from merlin.core import analysistask
from merlin.util import fixedpattern
from merlin.util import globalpositions

# Thresholds, from the LT074 samples quoted in RegistrationDiagnostics.
#: check 1: unrelated-FOV zero-shift peak / same-FOV peak above this
PATTERN_LOCK_RATIO = 0.5
#: check 2: a raw round shift below this (px) counts as locked
LOCKED_SHIFT_PX = 0.05
#: check 3: null bands whose median perpendicular shift is below this (px)
NULL_BAND_LOCK_PX = 1.0
#: check 3: below this real-edge band Pearson, beads cannot stitch
MIN_BAND_PEARSON = 0.05
#: check 4: held-out final error should be below this x affine-only
HELDOUT_RATIO = 0.5


def _phase_correlation(a: np.ndarray, b: np.ndarray):
    """(dy, dx, peak): b's shift relative to a as `FiducialCorrelationWarp`
    measures it (skimage, upsample 100), and the height of the normalised
    cross-power surface's maximum (0..1)."""
    shift = registration.phase_cross_correlation(a, b, upsample_factor=100)[0]
    crossPower = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    crossPower /= np.abs(crossPower) + 1e-12
    peak = float(np.real(np.fft.ifft2(crossPower)).max())
    return float(shift[0]), float(shift[1]), peak


def _band_pearson(a: np.ndarray, b: np.ndarray, dy: float, dx: float) -> float:
    """Pearson correlation of bands a and b over their overlap when b is
    moved by the registration shift (dy, dx), rounded to whole pixels."""
    oy, ox = int(round(dy)), int(round(dx))
    h, w = a.shape
    r0, r1 = max(0, oy), min(h, h + oy)
    c0, c1 = max(0, ox), min(w, w + ox)
    if r1 - r0 < 10 or c1 - c0 < 10:
        return np.nan
    aa = a[r0:r1, c0:c1].ravel()
    bb = b[r0 - oy:r1 - oy, c0 - ox:c1 - ox].ravel()
    if aa.std() == 0 or bb.std() == 0:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def _number(x) -> Optional[float]:
    """JSON-safe float (None for NaN)."""
    x = float(x)
    return None if np.isnan(x) else x


def _rounded(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(x, 3)


def _pearson(x, y) -> Optional[float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return _number(np.corrcoef(x, y)[0, 1])


class RegistrationDiagnostics(analysistask.AnalysisTask):

    """
    Checks, on a subset of `n_fovs` FOVs, whether the fiducial beads can
    register rounds (`FiducialCorrelationWarp`) and stitch FOVs
    (`RegisterFovNeighbors`), or whether the camera's fixed pattern
    dominates them. Run it before the full analysis.

    Writes a JSON report (`registration_diagnostics`), the per-FOV and
    per-band measurements (CSV) and a figure. The report ends with a
    recommended parameter block for both tasks, each choice explained by
    the check that drove it. It reports, recommends and warns; no task
    reads it. `fail_on_warning` raises instead of warning.

    The camera template is built here in memory, as `warp.FiducialTemplate`
    builds it, from `n_template_fovs` FOVs outside the subset. Images are
    oriented, hot pixels removed, and (checks 1-2) filtered by the
    `warp_task`'s `_filter` if that task is saved in this dataset, else by
    the default one.

    Checks (thresholds from LT074, 2026-09-27):

    1. Fixed-pattern lock, rounds: the reference fiducial of FOV a vs the
       comparison round's fiducial of an unrelated FOV b. A zero-shift peak
       as high as the same-FOV peak (> `PATTERN_LOCK_RATIO`) means the
       pattern, not the sample, drives the registration (BC555 disk raw:
       0.171 vs 0.176; after the template: 0.003). Recommends the template.
    2. First vs last bead frame (`second_fiducial_frame`, default the
       file's last frame; None skips): per FOV, reference -> comparison
       round, raw and with the template. Reports median |first - last|
       (strong beads 0.008-0.013 um, weak + template 0.012-0.019 um), the
       correlation across FOVs (> 0.98 when it works) and the fraction of
       raw shifts below `LOCKED_SHIFT_PX`.
    3. Stitching null bands: facing bands of two FOVs two steps apart (no
       overlap), Hann-windowed. A median perpendicular shift below
       `NULL_BAND_LOCK_PX`, in either direction, means the row pattern
       locks the bands (raw disk 0.0-0.1 px vs > 70 px elsewhere). Also
       the band Pearson of real
       edges after the template (BC555 disk ~0.02, BC553 disk 0.19-0.27,
       strong beads 0.5-0.9): below `MIN_BAND_PEARSON`, recommends
       stitching on `max_projection_data_channel`.
    4. If `global_alignment_task` has a `correction_summary`: warns if its
       rotation is more than `rotation_tolerance_deg` from
       `expected_rotation_deg` (ST2 -0.27 to -0.30, MF3 -0.94, MFX +0.01;
       BC555 disk's failed beads gave -0.15), or if its held-out final
       error is not below `HELDOUT_RATIO` x the affine-only error.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        defaults = {
            'n_fovs': 30,
            'n_template_fovs': 15,
            'seed': 0,
            'fiducial_data_channel': 0,
            # None: the first data channel with a different fiducial image
            'comparison_data_channel': None,
            'second_fiducial_frame': 'last',
            'max_projection_data_channel': 'DAPI',
            'warp_task': 'FiducialCorrelationWarp',
            'global_alignment_task': 'LeastSquaresGlobalAlignment',
            'expected_rotation_deg': None,
            'rotation_tolerance_deg': 0.05,
            'fail_on_warning': False,
        }
        for key, value in defaults.items():
            if key not in self.parameters:
                self.parameters[key] = value

    def get_estimated_memory(self):
        return 4000

    def get_estimated_time(self):
        return 60

    def get_dependencies(self):
        return []

    def _channel(self, channel) -> int:
        if isinstance(channel, str):
            return self.dataSet.get_data_organization().get_data_channel_index(
                channel)
        return int(channel)

    def _comparison_channel(self, referenceChannel: int) -> int:
        if self.parameters['comparison_data_channel'] is not None:
            return self._channel(self.parameters['comparison_data_channel'])
        dataOrganization = self.dataSet.get_data_organization()
        referenceImage = fixedpattern.fiducial_source(
            dataOrganization, referenceChannel)[:2]
        for channel in dataOrganization.get_data_channels():
            if fixedpattern.fiducial_source(
                    dataOrganization, channel)[:2] != referenceImage:
                return int(channel)
        raise ValueError('every data channel shares the reference fiducial '
                         'image; there is no round to register')

    def _filter_task(self):
        name = self.parameters['warp_task']
        if name is not None and self.dataSet.analysis_exists(name):
            return self.dataSet.load_analysis_task(name)
        return warp.FiducialCorrelationWarp(self.dataSet, {})

    def _run_analysis(self):
        fovs = np.asarray(self.dataSet.get_fovs())
        rng = np.random.default_rng(self.parameters['seed'])
        subset = np.sort(rng.choice(
            fovs, size=min(self.parameters['n_fovs'], len(fovs)), replace=False))
        referenceChannel = self._channel(self.parameters['fiducial_data_channel'])
        comparisonChannel = self._comparison_channel(referenceChannel)
        secondFrame = self.parameters['second_fiducial_frame']
        frames = [None] if secondFrame is None else [None, secondFrame]
        micronsPerPixel = self.dataSet.get_microns_per_pixel()

        def load(channel, fov, frame=None):
            return fixedpattern.load_fiducial_frame(
                self.dataSet, channel, int(fov), frame)

        templateFovs = fixedpattern.choose_template_fovs(
            fovs, self.parameters['n_template_fovs'],
            self.parameters['seed'] + 1, exclude=subset)
        templates = {
            (channel, frame): fixedpattern.median_template(
                [load(channel, f, frame) for f in templateFovs])
            for channel in (referenceChannel, comparisonChannel)
            for frame in frames}

        filterImage = self._filter_task()._filter
        _, positions, _, _, overlapFraction = \
            globalalign._nominal_positions_and_overlap(self.dataSet, None)

        roundRows, bandRows = [], []
        for i, fov in enumerate(subset):
            images = {(c, fr): load(c, fov, fr)
                      for c in (referenceChannel, comparisonChannel)
                      for fr in frames}
            unrelatedFov = subset[(i + 1) % len(subset)]
            unrelated = load(comparisonChannel, unrelatedFov)
            for variant in ('raw', 'template'):
                def prepared(image, channel, frame=None):
                    if variant == 'template':
                        image = image - templates[(channel, frame)]
                    return filterImage(image)

                reference = prepared(images[(referenceChannel, None)],
                                     referenceChannel)
                same = _phase_correlation(reference, prepared(
                    images[(comparisonChannel, None)], comparisonChannel))
                other = _phase_correlation(reference, prepared(
                    unrelated, comparisonChannel))
                row = {'fov': int(fov), 'unrelated_fov': int(unrelatedFov),
                       'variant': variant,
                       'first_dy_px': same[0], 'first_dx_px': same[1],
                       'same_peak': same[2],
                       'unrelated_dy_px': other[0], 'unrelated_dx_px': other[1],
                       'unrelated_peak': other[2]}
                if secondFrame is not None:
                    last = _phase_correlation(
                        prepared(images[(referenceChannel, secondFrame)],
                                 referenceChannel, secondFrame),
                        prepared(images[(comparisonChannel, secondFrame)],
                                 comparisonChannel, secondFrame))
                    row.update({'last_dy_px': last[0], 'last_dx_px': last[1]})
                roundRows.append(row)

            bandRows.extend(self._band_rows(
                int(fov), images[(referenceChannel, None)],
                templates[(referenceChannel, None)], referenceChannel,
                positions, overlapFraction, micronsPerPixel))

        rounds = pd.DataFrame(roundRows)
        bands = pd.DataFrame(bandRows, columns=[
            'anchor_fov', 'neighbor_fov', 'direction', 'kind', 'variant',
            'dy_px', 'dx_px', 'perpendicular_px', 'pearson'])
        self.dataSet.save_dataframe_to_csv(
            rounds, 'round_measurements', self, index=False)
        self.dataSet.save_dataframe_to_csv(
            bands, 'band_measurements', self, index=False)

        report = {
            'fovs': [int(f) for f in subset],
            'template_fovs': [int(f) for f in templateFovs],
            'fiducial_data_channel': referenceChannel,
            'comparison_data_channel': comparisonChannel,
            'microns_per_pixel': micronsPerPixel,
            'check1_pattern_lock': self._check1(rounds),
            'check2_first_vs_last': self._check2(rounds, micronsPerPixel),
            'check3_stitching_bands': self._check3(bands),
            'check4_global_alignment': self._check4(),
        }
        report['recommendation'] = self._recommend(report)
        report['warnings'] = [
            report[k]['warning'] for k in sorted(report)
            if k.startswith('check') and report[k].get('warning')]
        self.dataSet.save_json_analysis_result(
            report, 'registration_diagnostics', self.analysisName)
        self._save_figure(rounds, bands, templates[(referenceChannel, None)],
                          micronsPerPixel)

        for message in report['warnings']:
            if self.parameters['fail_on_warning']:
                raise RuntimeError('RegistrationDiagnostics: ' + message)
            warnings.warn(message)

    def _band_rows(self, anchorFov, anchorImage, template, channel,
                   positions, overlapFraction, micronsPerPixel) -> List[Dict]:
        """Check 3's measurements for one anchor: its +x and +y real edge,
        and the null pair two steps away in each direction."""
        rows = []
        for direction, dx, dy in (('+x', 1.0, 0.0), ('+y', 0.0, 1.0)):
            neighbor = globalpositions.find_grid_neighbor(
                anchorFov, positions, dx, dy)
            if neighbor is None or not globalpositions._within_registration_range(
                    anchorImage.shape, positions[anchorFov], positions[neighbor],
                    dx, dy, overlapFraction, micronsPerPixel):
                continue
            beyond = globalpositions.find_grid_neighbor(
                neighbor, positions, dx, dy)
            partners = [('real', neighbor)] + (
                [('no_overlap', beyond)] if beyond is not None else [])
            for kind, partnerFov in partners:
                partner = fixedpattern.load_fiducial_frame(
                    self.dataSet, channel, partnerFov)
                for variant in ('raw', 'template'):
                    a, b = anchorImage, partner
                    if variant == 'template':
                        a, b = a - template, b - template
                    bandA, bandB = globalpositions.crop_overlap(
                        a, b, dx, dy, overlapFraction)
                    shiftY, shiftX, _ = _phase_correlation(
                        globalpositions._hann_windowed(bandA),
                        globalpositions._hann_windowed(bandB))
                    rows.append({
                        'anchor_fov': anchorFov, 'neighbor_fov': int(partnerFov),
                        'direction': direction, 'kind': kind,
                        'variant': variant, 'dy_px': shiftY, 'dx_px': shiftX,
                        'perpendicular_px': shiftY if dx != 0 else shiftX,
                        'pearson': _band_pearson(bandA, bandB, shiftY, shiftX)})
        return rows

    @staticmethod
    def _check1(rounds: pd.DataFrame) -> Dict:
        out = {}
        for variant, g in rounds.groupby('variant'):
            atZero = np.hypot(g.unrelated_dy_px, g.unrelated_dx_px) < 1
            out[variant] = {
                'median_same_fov_peak': _number(g.same_peak.median()),
                'median_unrelated_fov_peak': _number(g.unrelated_peak.median()),
                'unrelated_fraction_at_zero_shift': _number(atZero.mean()),
            }
        raw = out['raw']
        ratio = raw['median_unrelated_fov_peak'] / raw['median_same_fov_peak']
        out['raw_unrelated_to_same_peak_ratio'] = _number(ratio)
        out['flagged'] = bool(ratio > PATTERN_LOCK_RATIO
                              and raw['unrelated_fraction_at_zero_shift'] >= 0.5)
        if out['flagged']:
            out['warning'] = (
                'check 1: the camera pattern drives round registration: an '
                'unrelated FOV correlates at zero shift with peak %.3f vs '
                '%.3f for the same FOV (%.3f after the template)' % (
                    raw['median_unrelated_fov_peak'],
                    raw['median_same_fov_peak'],
                    out['template']['median_unrelated_fov_peak']))
        return out

    @staticmethod
    def _check2(rounds: pd.DataFrame, micronsPerPixel: float) -> Dict:
        if 'last_dy_px' not in rounds:
            return {'skipped': 'second_fiducial_frame is None'}
        out = {}
        for variant, g in rounds.groupby('variant'):
            first = g[['first_dy_px', 'first_dx_px']].to_numpy() * micronsPerPixel
            last = g[['last_dy_px', 'last_dx_px']].to_numpy() * micronsPerPixel
            out[variant] = {
                'median_first_last_difference_um': _number(
                    np.median(np.hypot(*(first - last).T))),
                'pearson_first_last_y': _pearson(first[:, 0], last[:, 0]),
                'pearson_first_last_x': _pearson(first[:, 1], last[:, 1]),
                'median_shift_um': _number(np.median(np.hypot(*first.T))),
                'fraction_locked': _number(np.mean(
                    np.hypot(*first.T) / micronsPerPixel < LOCKED_SHIFT_PX)),
            }
        return out

    @staticmethod
    def _check3(bands: pd.DataFrame) -> Dict:
        out = {}
        for variant in ('raw', 'template'):
            g = bands[bands.variant == variant]
            null, real = g[g.kind == 'no_overlap'], g[g.kind == 'real']
            out[variant] = {
                'n_null_pairs': int(len(null)),
                # the row pattern locks one band orientation only
                'null_median_abs_perpendicular_px': {
                    direction: _number(d.perpendicular_px.abs().median())
                    for direction, d in null.groupby('direction')},
                'n_real_edges': int(len(real)),
                'real_median_pearson': _number(real.pearson.median()),
            }
        nullShifts = out['raw']['null_median_abs_perpendicular_px']
        lockedDirection = min(nullShifts, key=nullShifts.get) if nullShifts else None
        nullShift = nullShifts.get(lockedDirection)
        out['flagged'] = bool(nullShift is not None and nullShift < NULL_BAND_LOCK_PX)
        if out['flagged']:
            out['warning'] = (
                'check 3: the camera row pattern locks stitching bands: '
                'non-overlapping %s bands register at a median perpendicular '
                'shift of %.2f px; real-edge band Pearson after the template '
                '%s' % (lockedDirection, nullShift, _rounded(out['template']['real_median_pearson'])))
        return out

    def _check4(self) -> Dict:
        name = self.parameters['global_alignment_task']
        try:
            summary = self.dataSet.load_json_analysis_result(
                'correction_summary', name)
        except (OSError, TypeError):
            return {'available': False}
        out = {'available': True,
               'affine_rotation_deg': summary.get('affine_rotation_deg'),
               'heldout_edge_error_um': summary.get('heldout_edge_error_um')}
        problems = []
        expected = self.parameters['expected_rotation_deg']
        rotation = out['affine_rotation_deg']
        if expected is not None and rotation is not None and \
                abs(rotation - expected) > self.parameters['rotation_tolerance_deg']:
            problems.append('rotation %.3f deg, expected %.3f' % (rotation, expected))
        heldOut = out['heldout_edge_error_um'] or {}
        final, affineOnly = heldOut.get('final'), heldOut.get('affine_only')
        if final and affineOnly and \
                final['median'] > HELDOUT_RATIO * affineOnly['median']:
            problems.append('held-out final %.3f um vs affine only %.3f um' % (
                final['median'], affineOnly['median']))
        out['flagged'] = bool(problems)
        if problems:
            out['warning'] = 'check 4: %s looks failed: %s' % (
                name, '; '.join(problems))
        return out

    def _recommend(self, report: Dict) -> Dict:
        check1, check2 = report['check1_pattern_lock'], report['check2_first_vs_last']
        check3 = report['check3_stitching_bands']
        warpBlock, warpReasons = {}, []
        if check1['flagged']:
            warpBlock = {'remove_hot_pixels': True,
                         'fiducial_template_task': 'FiducialTemplate'}
            warpReasons.append('check 1 flagged: subtract the camera template')
        else:
            warpReasons.append('check 1 passed: raw beads are not '
                               'pattern-locked; no template needed')
        if 'template' in check2:
            variant = 'template' if check1['flagged'] else 'raw'
            r = [_rounded(check2[variant]['pearson_first_last_y']),
                 _rounded(check2[variant]['pearson_first_last_x'])]
            if any(x is None or x < 0.95 for x in r):
                warpReasons.append(
                    'check 2: first and last bead frames disagree (%s r = %s): '
                    'bead round registration is not confirmed' % (variant, r))
            else:
                warpReasons.append('check 2: first and last bead frames agree '
                                   '(%s r = %s)' % (variant, r))

        stitchBlock, stitchReasons = {}, []
        pearson = check3['template']['real_median_pearson']
        if pearson is None or pearson < MIN_BAND_PEARSON:
            stitchBlock = {'max_projection_data_channel':
                           self.parameters['max_projection_data_channel'],
                           'hann_window': True}
            stitchReasons.append(
                'check 3: real-edge band Pearson after the template %s < %s: '
                'the bands hold too little bead signal; stitch on a max '
                'projection' % (_rounded(pearson), MIN_BAND_PEARSON))
        elif check3['flagged']:
            stitchBlock = {'fiducial_template_task': 'FiducialTemplate'}
            stitchReasons.append('check 3 flagged, bead bands carry signal '
                                 '(Pearson %.2f): subtract the template' % pearson)
        else:
            stitchReasons.append('check 3 passed: stitch on the beads '
                                 '(Pearson %.2f)' % pearson)
        return {'FiducialCorrelationWarp': warpBlock,
                'FiducialCorrelationWarp_reasons': warpReasons,
                'RegisterFovNeighbors': stitchBlock,
                'RegisterFovNeighbors_reasons': stitchReasons}

    def _save_figure(self, rounds, bands, template, micronsPerPixel):
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        low, high = np.percentile(template, [1, 99])
        axes[0].imshow(template, cmap='gray', vmin=low, vmax=high)
        axes[0].set_title('reference-round camera template')
        if 'last_dy_px' in rounds:
            for (variant, g), marker in zip(rounds.groupby('variant'), 'os'):
                for axis, color in (('dy', '#eb6834'), ('dx', '#2a78d6')):
                    axes[1].scatter(g['first_%s_px' % axis] * micronsPerPixel,
                                    g['last_%s_px' % axis] * micronsPerPixel,
                                    s=20, marker=marker, color=color, alpha=0.7,
                                    label='%s %s' % (variant, axis))
            axes[1].axline((0, 0), slope=1, color='k', lw=1, ls='--')
            axes[1].legend()
        axes[1].set_xlabel('shift from first bead frames (um)')
        axes[1].set_ylabel('shift from last bead frames (um)')
        axes[1].set_title('check 2: round shift, measured twice')
        for x, (variant, kind) in enumerate(
                [(v, k) for v in ('raw', 'template') for k in ('no_overlap', 'real')]):
            g = bands[(bands.variant == variant) & (bands.kind == kind)]
            axes[2].scatter(np.full(len(g), x) + np.random.default_rng(0).uniform(
                -0.2, 0.2, len(g)), g.perpendicular_px.abs().clip(lower=1e-2),
                s=12, alpha=0.7)
        axes[2].set_yscale('log')
        axes[2].set_xticks(range(4))
        axes[2].set_xticklabels(['raw null', 'raw real', 'template null',
                                 'template real'])
        axes[2].axhline(NULL_BAND_LOCK_PX, color='#c0392b', ls='--', lw=1)
        axes[2].set_ylabel('|perpendicular band shift| (px)')
        axes[2].set_title('check 3: stitching bands')
        fig.tight_layout()
        self.dataSet.save_figure(self, fig, 'registration_diagnostics')
        plt.close(fig)
