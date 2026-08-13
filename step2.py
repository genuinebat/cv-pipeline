import os
from label import AutoLabeler

if __name__ == "__main__":
    INPUT = "./2_input_imgs"
    VALIDATED = "./4_validated_imgs"
    LABELED = "./3_labled_imgs"
    UNLABELDED = "./3_unlabeled_imgs"

    ontology = {
        "caption": "class",
    }

    al = AutoLabeler(ont=ontology, input_dir=INPUT, validated_dir=VALIDATED, labeled_dir=LABELED, unlabeled_dir=UNLABELDED)

    al.label_and_validate()
    
    labeled_validated= len(os.listdir(al.VALIDATED_DIR))
    labeled = len(os.listdir(al.LABELED_DIR))
    unlabled = len(os.listdir(al.UNLABELED_DIR))
    total = labeled_validated+labeled+unlabled

    print(f"""
    ------------ Phase 2 Results ------------
    {total} total images processed

    {labeled_validated} image{"" if labeled_validated == 1 else "s"} were labeled and passed secondary validation
    {labeled} image{"" if labeled == 1 else "s"} were labeled but failed secondary validation
    {unlabled} image{"" if unlabled == 1 else "s"} could not be labeled
    -----------------------------------------
    """)