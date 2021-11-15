## Классификация звезда/галактика/квазар с помощью технологии DVC

### DVC Cache
Для хранения и версионирования файлов в DVC используется *content-based* хранилище. 
Содержание (или контент) файла однозначно определяется *хэш-суммой* — последовательностью из 32 шестандцатиричных символов (по дефолту используется алгоритм MD5). 
Контенты файлов хранятся в т.н. кэше DVC (по дефолту директория **[.dvc/cache](.dvc/cache)**): первые 2 символа используются для именования директории в кэше, а остальная часть — для названия непосредственно файла с содержанием. 
Таким образом, при добавлении файла в «индекс» DVC: **`$ dvc add <file>`** для него вычисляется хэш, который указывается в файле \<file\>.dvc, также в файле \<file\>.dvc указывается размер файла \<file\>. 
При этом файл \<file\> автоматически добавляется в .gitignore и не индексируется Git’ом, индексируется только «ссылка» на него — \<file\>.dvc. 
При такой реализации хранения файлы с разными именами, но одинаковым содержанием не будут дублироваться в кэше.

### Версионирование пайплайнов
Каждый раз при воспроизведении пайплайна командой **`$ dvc repro`** вычисляются хэш-суммы deps и outs файлов, указанных в описании пайплайна (файл **[dvc.yaml](dvc.yaml)**), которые сравниваются с хэш-суммами в описании пайплайна при предыдущем запуске (файл **[dvc.lock](dvc.lock)**). 
Если хэши файлов изменились, то соответствующие и зависимые от них этапы пересчитываются. 
Также этапы пайплайна пересчитываются, если изменились команды для вычисления или параметры (файл **[params.yaml](params.yaml)**) этапов.

### Удалённое хранилище
DVC позволяет использовать удалённые хранилища для хранения кэша проекта. Например, для использования Google Drive в качестве удалённого хранилища необходимо выполнить: \
**`$ dvc remote add -d storage gdrive://<token from Google Drive URL>`** \
Для выгрузки локального кэша в удалённое хранилище необходимо выполнить **`$ dvc push`**, для загрузки из удалённого хранилища — **`$ dvc pull`**. 
Если файлы добавляются в проект через **`$ dvc pull`** то видимые в рабочем пространстве файлы являются ссылками (а не копиями) на соответствующие файлы в кэше.

### CI/CD с помощью GitHub Actions и CML
С помощью инструмента GitHub Actions можно осуществлять тестирование пайплайна при каждом коммите в GItHub. 
Для этого необходимо прописать нужные команды воспроизведения пайплайна в файл **[.github/workflows/update_project.yaml](.github/workflows/update_project.yaml)** (название опционально):
```
run: |
  pip install -r requirements.txt
  dvc pull
  echo "dvc pull completed sucessfully" > report.md
  dvc repro >> report.md
  echo "dvc repro completed sucessfully" >> report.md
  dvc metrics show -A >> report.md
  cml-send-comment report.md
```
При этом если кэш проекта хранится в Google Drive, при загрузке файлов в тестовое пространство GitHub Actions командой \
**`$ dvc pull`** потребуется аутентификация. 
Для автоматизации аутентификации необходимо указать реквизиты — содержание файла **[.dvc/tmp/gdrive-user-credentials.json](.dvc/tmp/gdrive-user-credentials.json)**, автоматически сформированного при первом доступе к хранилищу, в т. н. Secrets репозитория (Settings -> Secrets), а затем добавить этот Secret в настройки тестового окружения:
```
env: 
	GDRIVE_CREDENTIALS_DATA: ${{ secrets.GOOGLE_DRIVE_STORAGE }}
```
CML же позволяет формировать и публиковать результаты отчета (например, в виде Markdown файла) как комментарий к соответствующему коммиту.
Чтобы без дополнительной установки использовать DVC и CML в тестовом пространстве необходимо указать это в настройках тестового пространства:
```
steps:
	- uses: actions/checkout@v2
	- uses: iterative/setup-dvc@v1
	- uses: iterative/setup-cml@v1
```
