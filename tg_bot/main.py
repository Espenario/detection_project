import telebot
import os
import sys
import cv2

sys.path.insert(1, "fastercnn-pytorch-training-pipeline")
import inference_bot

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

bot = telebot.TeleBot(os.environ['BOT_TAG'])

@bot.message_handler(content_types=['photo'])
def get_photo(message):
    bot.reply_to(message, 'Результат')
    # bot.send_photo(message.chat.id, photo=open('buff.jpg', 'rb'), reply_to_message_id = message.id)
    file_info = bot.get_file(message.photo[0].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    # bot.reply_to(message, file_info.file_path)
    src = 'buff.jpg'
    print(type(message.photo[0]))
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)
    params = {'input': 'buff.jpg', 'data': None, 'model': None,
      'weights': 'detection_stuff/outputs/training/fasterrcnn_resnet50_fpn_v2_trainaug_30e/best_model.pth',
       'threshold': 0.8, 'show': False, 'mpl_show': False, 'device':
        'cuda', 'imgsz': 640, 'no_labels': False, 'square_img': False}
    image_lsit = inference_bot.main(params)
    # bot.reply_to(message, 'Результат2')
    for image in image_lsit:
        cv2.imwrite(f"buff1.jpg", image)
        bot.send_photo(message.chat.id, photo=open('buff1.jpg', 'rb'), reply_to_message_id = message.id)


@bot.message_handler(commands=['start'])
def start_func(message):
    bot.send_message(message.chat.id, 'Привет!')

bot.polling(none_stop=True) 