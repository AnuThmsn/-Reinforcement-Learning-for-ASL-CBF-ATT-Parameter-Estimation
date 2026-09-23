import nibabel as nib
import os, json, sys

DATA = r"D:\reinforcement learning\asl-github-repo\test data"
for f in sorted(os.listdir(DATA)):
    if f.endswith(".nii") or f.endswith(".nii.gz"):
        path = os.path.join(DATA, f)
        img = nib.load(path)
        hdr = img.header
        print(f"=== {f} ===")
        print(f"  Shape: {img.shape}")
        pixdim = list(hdr["pixdim"])
        print(f"  Pixdim: {pixdim}")
        descrip = str(hdr["descrip"])
        print(f"  Descrip: {descrip}")
        print(f"  Intent code: {hdr['intent_code']}")
        print(f"  qform_code: {hdr['qform_code']}")
        print(f"  sform_code: {hdr['sform_code']}")
        # Print full header for timing fields
        print(f"  toffset: {hdr['toffset']}")
        print(f"  slice_duration: {hdr['slice_duration']}")
        print(f"  xyzt_units: {hdr['xyzt_units']}")
        # Check extensions
        if hasattr(img.header, 'extensions') and img.header.extensions:
            for ext in img.header.extensions:
                code = ext.get_code()
                content = ext.get_content()
                print(f"  Extension code={code}, size={len(content)}")
                try:
                    txt = content.decode("utf-8", errors="replace")[:500]
                    print(f"  Content preview: {txt}")
                except:
                    pass
        print()

