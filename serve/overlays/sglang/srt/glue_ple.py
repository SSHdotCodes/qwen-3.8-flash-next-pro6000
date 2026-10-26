"""Optional exact-BF16 PLE small-M tactics; measured gain is only about 1.5 us/step."""
import os


def install_ple():
    if os.environ.get('QWENFAST','1')!='1' or os.environ.get('QWENFAST_PLE_BF16','0')!='1':return
    from sglang.srt import qwenfast
    # These two BF16 weight shapes are PLE layer 1's key and value projections.
    # Other row counts keep the original method, including M=2,3,5..8.
    qwenfast.SMALLM_TABLE[(10240,2560)]={1:(2,2),4:(4,2)}
    qwenfast.SMALLM_TABLE[(2560,2560)]={1:(2,2),4:(4,2)}
