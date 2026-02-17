import os
import io

from fastapi import FastAPI, Request
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types.input_file import BufferedInputFile

from retouch import retouch_image_bytes, PRESETS
from collage import make_before_after_collage

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN env var is required")

app = FastAPI()
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

START_TEXT = (
    "Привет! Я делаю натуральную ретушь (лицо + шея + декольте) без изменения черт лица.\n"
    "✅ Ресницы/глаза не трогаю. Кожа остаётся живой, без “пластика”.\n\n"
    "⚠️ Важно для максимального качества: отправляй фото только как **Файл (Документ)** — так Telegram не сжимает изображение.\n"
    "Как отправить:\n"
    "1) Нажми 📎 (скрепка)\n"
    "2) Выбери **Файл / Документ**\n"
    "3) Выбери фото и отправь\n\n"
    "Жду фото как документ 👇"
)

PHOTO_REJECT_TEXT = (
    "Это отправлено как **Фото**, Telegram его сжимает.\n"
    "Пожалуйста, отправь это же изображение **как Файл (Документ)**:\n"
    "📎 → Файл/Документ → выбрать фото → отправить."
)

NOT_IMAGE_TEXT = "Похоже, это не фото-файл. Пришли, пожалуйста, изображение (.JPG/.PNG/.HEIC) как **Документ**."
CHOOSE_MODE_TEXT = "Фото получено ✅ Выбери режим обработки:"


# MVP storage in memory: user_id -> bytes
USER_LAST: dict[int, bytes] = {}


def kb_modes():
    kb = InlineKeyboardBuilder()
    kb.button(text="🌿 Натурально", callback_data="mode:natural")
    kb.button(text="✨ Чище кожа", callback_data="mode:clean")
    kb.button(text="🔆 Только убрать блеск", callback_data="mode:shine")
    kb.adjust(1)
    return kb.as_markup()


@dp.message(CommandStart())
async def start(message: Message):
    await message.answer(START_TEXT, parse_mode="Markdown")


@dp.message(F.photo)
async def reject_photo(message: Message):
    await message.answer(PHOTO_REJECT_TEXT, parse_mode="Markdown")


@dp.message(F.document)
async def on_document(message: Message):
    doc = message.document
    filename = (doc.file_name or "").lower()
    mime = (doc.mime_type or "").lower()
    is_image = mime.startswith("image/") or filename.endswith((".jpg", ".jpeg", ".png", ".heic", ".webp"))

    if not is_image:
        await message.answer(NOT_IMAGE_TEXT, parse_mode="Markdown")
        return

    # Download file bytes
    file = await bot.get_file(doc.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    data = buf.getvalue()

    USER_LAST[message.from_user.id] = data
    await message.answer(CHOOSE_MODE_TEXT, reply_markup=kb_modes())


@dp.callback_query(F.data.startswith("mode:"))
async def process_mode(cb: CallbackQuery):
    await cb.answer()

    user_id = cb.from_user.id
    data = USER_LAST.get(user_id)
    if not data:
        await cb.message.answer("Сначала отправь фото как **Документ** 👇", parse_mode="Markdown")
        return

    mode = cb.data.split(":", 1)[1].strip()
    if mode not in PRESETS:
        mode = "natural"

    await cb.message.answer(
        f"Обрабатываю: **{PRESETS[mode].name}** ✨\n(максимальное качество, может занять немного времени)",
        parse_mode="Markdown",
    )

    after_jpeg, before_jpeg = retouch_image_bytes(data, mode)
    collage_jpeg = make_before_after_collage(before_jpeg, after_jpeg)

    ret_file = BufferedInputFile(after_jpeg, filename="retouched.jpg")
    col_file = BufferedInputFile(collage_jpeg, filename="before_after.jpg")

    await cb.message.answer_document(ret_file, caption="Готово ✅ Ретушь (файл без сжатия).")
    await cb.message.answer_document(col_file, caption="Коллаж До/После ✅ (файл без сжатия).")


@app.post("/webhook")
async def telegram_webhook(request: Request):
    update = await request.json()
    await dp.feed_webhook_update(bot, update)
    return {"ok": True}


@app.get("/")
async def root():
    return {"status": "ok"}
