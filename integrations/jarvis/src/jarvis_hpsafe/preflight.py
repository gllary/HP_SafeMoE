"""Validate input artifacts and lock this package before proposal fitting."""
from .common import ROOT,atomic_json,source_hashes
from .inputs import Inputs
from .train import canary

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'stage2/selection_locked.json').exists(),'Use a new output directory for a new fit'
    x=Inputs();assert len(x.tasks)==34
    canary()
    atomic_json(ROOT/'status/source_lock.json',source_hashes())
    atomic_json(ROOT/'status/preflight.json',{'passed':True,'source_stage1_fits':204,'test_labels_read':False})
if __name__=='__main__':main()
