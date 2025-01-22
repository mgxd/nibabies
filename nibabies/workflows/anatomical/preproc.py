import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from niworkflows.engine.workflows import LiterateWorkflow


def init_anat_preproc_wf(
    *,
    bspline_fitting_distance: int = 200,
    clipping: bool = False,
    name: str = 'anat_preproc_wf',
) -> LiterateWorkflow:
    """
    This workflow accepts T1w/T2w images as inputs (either raw or a merged template) and performs:
    - N4 Bias Field Correction
    - Intensity Clipping (optional)

    The outputs of this workflow will be a structural reference used for subsequent processing.

    Inputs
    ------
    in_anat : :obj:`str`
        A single denoised volume T1w/T2w image.
        If multiple runs were found, this is a merged template.
    clipping : :obj:`bool`
        Perform intensity clipping prior & after N4 bias correction (default: False)

    Outputs
    -------
    anat_preproc: :obj:`str`
        Preprocessed anatomical image (Denoising/INU/Clipping)
    """
    from nipype.interfaces.ants import N4BiasFieldCorrection
    from niworkflows.interfaces.header import ValidateImage
    from niworkflows.interfaces.nibabel import IntensityClip

    workflow = LiterateWorkflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['in_anat']),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['anat_preproc']),
        name='outputnode',
    )

    # validate image
    validate = pe.Node(ValidateImage(), name='anat_validate', run_without_submitting=True)
    n4_correct = pe.Node(
        N4BiasFieldCorrection(
            dimension=3,
            bspline_fitting_distance=bspline_fitting_distance,
            save_bias=True,
            copy_header=True,
            n_iterations=[50] * 5,
            convergence_threshold=1e-7,
            rescale_intensities=True,
            shrink_factor=4,
        ),
        name='n4_correct',
    )

    workflow.connect(inputnode, 'in_anat', validate, 'in_file')

    if clipping:
        clip_pre = pe.Node(IntensityClip(p_min=10.0, p_max=99.5), name='clip_pre_n4')
        clip_post = pe.Node(IntensityClip(p_min=5.0, p_max=99.5), name='clip_post_n4')

        workflow.connect([
            (validate, clip_pre, [('out_file', 'in_file')]),
            (clip_pre, n4_correct, [('out_file', 'input_image')]),
            (n4_correct, clip_post, [('output_image', 'in_file')]),
            (clip_post, outputnode, [('out_file', 'anat_preproc')]),
        ])  # fmt:skip

    else:
        workflow.connect([
            (validate, n4_correct, [('out_file', 'input_image')]),
            (n4_correct, outputnode, [('output_image', 'anat_preproc')]),
        ])  # fmt:skip

    return workflow


def init_csf_norm_wf(name: str = 'csf_norm_wf') -> LiterateWorkflow:
    """Replace low intensity voxels within the CSF mask with the median value."""

    workflow = LiterateWorkflow(name=name)
    workflow.__desc__ = (
        'The CSF mask was used to normalize the anatomical template by the median of voxels '
        'within the mask.'
    )
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['anat_preproc', 'anat_tpms']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['anat_preproc']), name='outputnode')

    # select CSF from BIDS-ordered list (GM, WM, CSF)
    select_csf = pe.Node(niu.Select(index=2), name='select_csf')
    norm_csf = pe.Node(niu.Function(function=_normalize_roi), name='norm_csf')

    workflow.connect([
        (inputnode, select_csf, [('anat_tpms', 'inlist')]),
        (select_csf, norm_csf, [('out', 'mask_file')]),
        (inputnode, norm_csf, [('anat_preproc', 'in_file')]),
        (norm_csf, outputnode, [('out', 'anat_preproc')]),
    ])  # fmt:skip

    return workflow


def _normalize_roi(in_file, mask_file, threshold=0.2, out_file=None):
    """Normalize low intensity voxels that fall within a given mask."""
    import nibabel as nb
    import numpy as np

    img = nb.load(in_file)
    img_data = np.asanyarray(img.dataobj)
    mask_img = nb.load(mask_file)
    # binary mask
    bin_mask = np.asanyarray(mask_img.dataobj) > threshold
    mask_data = bin_mask * img_data
    masked_data = mask_data[mask_data > 0]

    median = np.median(masked_data).astype(masked_data.dtype)
    normed_data = np.maximum(img_data, bin_mask * median)

    oimg = img.__class__(normed_data, img.affine, img.header)
    if not out_file:
        from nipype.utils.filemanip import fname_presuffix

        out_file = fname_presuffix(in_file, suffix='normed')
    oimg.to_filename(out_file)
    return out_file
