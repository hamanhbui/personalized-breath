import os
for i in range(100):
    os.system("python src/main.py --breath_type=normal --model_type=acce-gyro --model_name=tcn --no_outer=4")
    # os.system("python src/main.py --breath_type=normal --model_type=audio --model_name=tcn --no_outer=4")
    # os.system("python src/main.py --breath_type=normal --model_type=multi --model_name=tcn --no_outer=4")

    # os.system("python src/main.py --breath_type=normal --model_type=acce-gyro --model_name=cnn-lstm --no_outer=4")
    # os.system("python src/main.py --breath_type=normal --model_type=audio --model_name=cnn-lstm --no_outer=4")
    # os.system("python src/main.py --breath_type=normal --model_type=multi --model_name=cnn-lstm --no_outer=4")
