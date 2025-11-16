#Tensorflow Lite 실행 코드
import numpy as np
import os
from tflite_runtime.interpreter import Interpreter
from PIL import Image


MODEL_PATH = 'model.tflite'
NUM_TEST_IMAGES = 10 
IMAGE_PREFIX = 'sample_'



try:
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"모델 로드 또는 텐서 할당 중 오류 발생: {e}")
    exit()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'] # [1, 높이, 너비, 채널] 형태 예상
input_dtype = input_details[0]['dtype']

print(f"모델 입력 텐서 형태: {input_shape}")
print(f"모델 입력 데이터 타입: {input_dtype}")


TARGET_HEIGHT = input_shape[1]
TARGET_WIDTH = input_shape[2]


def run_inference_on_image(image_path, interpreter, input_details, output_details, target_h, target_w, input_dtype):
    


    try:
        image = Image.open(image_path).convert('L').resize((target_w, target_h))
    except FileNotFoundError:
        print(f"경고: 파일 {image_path}를 찾을 수 없습니다.")
        return None
    

    input_data = np.array(image, dtype=np.float32) # 일단 float32로 처리


    input_data = input_data / 255.0



    

    if len(input_shape) == 4 and input_shape[3] == 1: # [1, H, W, 1] 형태 (채널 1)
        input_data = np.expand_dims(input_data, axis=0) # [1, H, W]
        input_data = np.expand_dims(input_data, axis=-1) # [1, H, W, 1]
    elif len(input_shape) == 3: # [1, H, W] 형태
        input_data = np.expand_dims(input_data, axis=0)
    else:
        print("경고: 모델의 입력 형태가 예상과 다릅니다. [1, H, W, 1] 또는 [1, H, W]를 가정했습니다.")

   
    if input_dtype == np.uint8:
   
        input_data = (input_data * 255).astype(np.uint8)
    elif input_dtype == np.int8:
   
   
        pass

    
    interpreter.set_tensor(input_details[0]['index'], input_data)

    
    interpreter.invoke()

    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    
    
    prediction = np.argmax(output_data[0])
    
    return prediction, output_data[0]



print("\n--- MNIST 이미지 10장에 대한 추론 시작 ---")
results = {}


for i in range(NUM_TEST_IMAGES):
    image_filename = f"{IMAGE_PREFIX}{i}.png" # 파일명: mnist_0.png, mnist_1.png, ...
    

    if not os.path.exists(image_filename):
        print(f"🚫 파일 '{image_filename}'을 찾을 수 없습니다. 건너뜁니다.")
        continue


    result = run_inference_on_image(
        image_filename, 
        interpreter, 
        input_details, 
        output_details, 
        TARGET_HEIGHT, 
        TARGET_WIDTH,
        input_dtype
    )
    
    if result is not None:
        prediction, probabilities = result
        print(f"✅ 파일: {image_filename} | 예측된 숫자: **{prediction}** | 확률 (일부): {probabilities[:5]}")
        results[image_filename] = prediction


print("\n--- 전체 추론 완료 ---")
