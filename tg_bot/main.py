import asyncio
import os
import sys
import cv2
import logging
import glob

from aiogram import Bot, Dispatcher, types
# from aiogram.filters.command import Command
from aiogram.types import ContentType, Message, InputFile

sys.path.insert(1, "fastercnn-pytorch-training-pipeline")
import inference_bot

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(os.environ['BOT_TAG'])
# Диспетчер
dp = Dispatcher(bot)

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# @bot.message_handler(content_types=['photo'])
# def get_photo(message):
#     bot.reply_to(message, 'Результат')
#     # bot.send_photo(message.chat.id, photo=open('buff.jpg', 'rb'), reply_to_message_id = message.id)
#     file_info = bot.get_file(message.photo[0].file_id)
#     downloaded_file = bot.download_file(file_info.file_path)
#     # bot.reply_to(message, file_info.file_path)
#     src = 'buff.jpg'
#     print(type(message.photo[0]))
#     with open(src, 'wb') as new_file:
#         new_file.write(downloaded_file)
#     params = {'input': 'buff.jpg', 'data': None, 'model': None,
#       'weights': 'detection_stuff/outputs/training/fasterrcnn_resnet50_fpn_v2_trainaug_30e/best_model.pth',
#        'threshold': 0.8, 'show': False, 'mpl_show': False, 'device':
#         'cuda', 'imgsz': 640, 'no_labels': False, 'square_img': False}
#     image_lsit = inference_bot.main(params)
#     # bot.reply_to(message, 'Результат2')
#     for image in image_lsit:
#         cv2.imwrite(f"buff1.jpg", image)
#         bot.send_photo(message.chat.id, photo=open('buff1.jpg', 'rb'), reply_to_message_id = message.id)

@dp.message_handler(content_types=ContentType.PHOTO)
async def cmd_photo(message: Message):
    await message.photo[-1].download()
    # определяем путь к фото
    img_path = (await bot.get_file(message.photo[-1].file_id)).file_path
    params = {'input': img_path, 'data': None, 'model': None,
      'weights': 'detection_stuff/outputs/training/fasterrcnn_resnet50_fpn_v2_trainaug_30e/best_model.pth',
       'threshold': 0.8, 'show': False, 'mpl_show': False, 'device':
        'cuda', 'imgsz': 640, 'no_labels': False, 'square_img': False}
    image_lsit = inference_bot.main(params)
    for image in image_lsit:
        cv2.imwrite(f"buff.jpg", image)
        with open('buff.jpg', 'rb') as photo:
            chat_id = message.from_user.id
            await bot.send_photo(chat_id, photo)
            # bot.send_photo(message.chat.id, photo=open('buff1.jpg', 'rb'), reply_to_message_id = message.id)
    removing_files = glob.glob(img_path)
    for i in removing_files:
        os.remove(i)
# Хэндлер на команду /start
@dp.message_handler(text="/start")
async def cmd_start(message: types.Message):
    await message.answer("Привет!")

# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())