def dall_e_3_generate_image(emotion):
  client = OpenAI(api_key='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')
  response = client.images.generate(
    model="dall-e-3",
    # 寫實風格
    #prompt=emotion + " elderly portrait in a realistic style, focusing solely on the human head without background, and with low resolution in grayscale.",

    # 真人風格
    # prompt=emotion + " elderly portrait in a true-to-life style, focusing solely on the human head without background, and with low resolution in grayscale.",

    # 癡呆症老人 + 真人風格 + 亞洲臉孔 + 情緒
    prompt="a " + emotion +  " single-portrait image of an dementia displaying elderly male or female with an Asian facial appearance in a true-to-life style, focusing solely on the human head without background, and with low resolution in grayscale.",

    size="1024x1024",
    quality="standard",
    n=1,
  )
  image_url = response.data[0].url
  return image_url

import cv2
def resizeImage(image):
  # 將PIL Image轉換為OpenCV格式
  cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

  # 調整圖片大小
  cv_image = cv2.resize(cv_image, (48, 48))

  # 顯示圖片
#   cv2_imshow(cv_image) # show image for colab
  cv2.imshow('Image', cv_image)

  # 將OpenCV格式的圖像轉換回PIL Image
  resize_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

  return resize_image


import os
from datetime import datetime

def add_timestamp_to_filename(filename):
    #"""將日期和時間附加到檔案名稱"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name, extension = os.path.splitext(filename)
    new_filename = f"{base_name}_{timestamp}{extension}"
    return new_filename



def getFileName(emotion, folder_path="./"):
  folder_path = folder_path + emotion+"/"

  # 检查文件夹是否存在
  if not os.path.exists(folder_path):
      # 如果不存在，则创建文件夹
      os.makedirs(folder_path)
      print(f"文件夹 '{folder_path}' 已创建。")
  else:
      print(f"文件夹 '{folder_path}' 已经存在。")

  filename = add_timestamp_to_filename(folder_path + emotion + ".png")

  return filename


import numpy as np

def generate_avatar(emotion):
  # API endpoint
  api_url = dall_e_3_generate_image(emotion)
#   api_url = "https://thispersondoesnotexist.com"

  # 發送 GET 請求
  response = requests.get(api_url)

  # 檢查回應狀態碼D
  if response.status_code == 200:
      # 將二進制數據轉換為圖片
      image = Image.open(BytesIO(response.content))
      
      # 存原圖
      image.save(getFileName(emotion, "./original/")) 
      image.save("./original/photo.png") # 存方便看的
      image.save("photo.png") # 存給情緒辨識使用

      # 調整圖片大小
      resize_image = resizeImage(image)

      # 將圖片存檔
      resize_image.save(getFileName(emotion), "PNG")
      

  else:
      print(f"Error: {response.status_code}")


import requests
from PIL import Image
from io import BytesIO
from openai import OpenAI

if __name__ == "__main__":
    # emotion = "angry"
    emotion = "happy"
    # emotion = "sad"
    # emotion = "fear"
    # emotion = "disgust"
    # emotion = "surprise"
    # emotion = "a slightly annoyed or disgruntled expression"
    # emotion = "a slightly annoyed, disgruntled, or emotionally distressed expression"
    generate_avatar(emotion)