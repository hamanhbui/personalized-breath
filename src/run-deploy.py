import os
for i in range(100):
    os.system("python src/main.py --breath_type=deep --model_type=acce-gyro --model_name=tcn")
    # os.system("python src/main.py --breath_type=deep --model_type=audio --model_name=tcn")
    # os.system("python src/main.py --breath_type=deep --model_type=multi --model_name=tcn")

    # os.system("python src/main.py --breath_type=deep --model_type=acce-gyro --model_name=cnn-lstm")
    # os.system("python src/main.py --breath_type=deep --model_type=audio --model_name=cnn-lstm")
    # os.system("python src/main.py --breath_type=deep --model_type=multi --model_name=cnn-lstm")
