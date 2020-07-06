import os
for i in range(100):
    # os.system("python src/main.py --breath_type=strong --model_type=acce-gyro --model_name=tcn --no_outer=0")
    # os.system("python src/main.py --breath_type=strong --model_type=audio --model_name=tcn --no_outer=0")
    os.system("python src/main.py --breath_type=strong --model_type=multi --model_name=tcn --no_outer=0")

    # os.system("python src/main.py --breath_type=strong --model_type=acce-gyro --model_name=cnn-lstm --no_outer=0")
    # os.system("python src/main.py --breath_type=strong --model_type=audio --model_name=cnn-lstm --no_outer=0")
    # os.system("python src/main.py --breath_type=strong --model_type=multi --model_name=cnn-lstm --no_outer=0")
