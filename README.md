# personalized-breath
Repo for the ACMMM20 submission: ["Personalized breath based biometric authentication with wearable multimodality"](https://arxiv.org/abs/2110.15941).
```bibtex
@misc{bui2021personalized,
      title={Personalized breath based biometric authentication with wearable multimodality}, 
      author={Manh-Ha Bui and Viet-Anh Tran and Cuong Pham},
      year={2021},
      eprint={2110.15941},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
**Please CITE** our paper if you find it useful in your research.
### Guideline
To extract features:
- cd personalized-breath/
- python src/processing/extract_acce_gyro.py
- python src/processing/extract_audio.py

To train model:
- python src/main.py --breath_type=normal --model_type=multi --model_name=tcn --no_outer=0
- results stored in results/

To get results:
- with open('results/outputs/'+breath_type+'/list_test_acc_'+model_name, 'rb') as filehandle:
    list_test_acc = pickle.load(filehandle)
