# personalized-breath
Repo for the ACMMM20 submission: ["Personalized breath based biometric authentication with wearable multimodality"](https://drive.google.com/file/d/1Br19A6AT6fNyDWEIHElHaI89Bn7vgPNK/view?usp=sharing)

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
