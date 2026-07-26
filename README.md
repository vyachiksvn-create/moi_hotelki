# MOI HOTELKI
Автоматический пайплайн для сортировки фотографий по полу и поиска визуальных дубликатов лиц.

## Быстрый старт
1. Установите зависимости: `pip install -r requirements.txt`
2. Поместите фото в `C:\Foto\Baza`
3. Запустите: `python main.py --mode 4`

## Режимы
- `--mode 1` — анализ пола в `Baza`, мужчины -> `Parni`, битые/без лиц -> `Nea`
- `--mode 2` — поиск дубликатов в `Parni`
- `--mode 3` — поиск дубликатов в `Baza`
- `--mode 4` — полный автопайплайн: gender split + dupes in `Parni`
- `--reset` — вернуть всё из `Parni`, `Sovpadenia`, `Nea` обратно в `Baza`

## Важно
В данной конфигурации модель `buffalo_l` возвращает `gender=0` для мужчин и `gender=1` для женщин. Это учтено в `config.json` (`gender_mapping`).

## Конфигурация
- `config.json` — пороги, папки, `gender_mapping`, `performance.cpu_det_size`, `performance.gpu_det_size`
- `main.py` — основной авто-пайплайн
- `auto_find.py` — утилита поиска дублей `Baza -> Цифры` с ФИО/ФИ группировкой

## Кэш
- `.embeddings_cache.pkl` — кэш эмбеддингов в `C:\Foto`
- `app.log` — лог выполнения
