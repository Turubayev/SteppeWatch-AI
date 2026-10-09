# SteppeWatch-AI: проект и заявка в Claude Startups

Обновлено 9 октября 2026 года. Это подготовка заявки, не подтверждение участия.

## Продукт

SteppeWatch-AI — ранний проект спутникового мониторинга земель Казахстана. Он помогает находить участки для наземной проверки: изменения растительности с поправкой на осадки, возможное засоление и гари. Целевые пользователи — пастбищные хозяйства, экологические организации и специалисты по землепользованию. Они являются целевым сегментом, а не заявленными клиентами.

Гипотеза бизнеса: платные отчёты по выбранному району и регулярный мониторинг. Цена, спрос, действующие клиенты, выручка и регистрация компании пока не подтверждены в этом документе. Открытый код распространяется по Apache 2.0.

Существуют Python-пайплайн, локальный Streamlit-дашборд, синтетические тесты и сохранённый спутниковый пример Приаралья. Это исследовательский прототип. Полевой проверки качества и сертифицированного отчёта UNCCD нет. Актуальная сводка — `results/kyzylorda.json`, ревизия 2; прежние цифры README нельзя использовать как текущие результаты.

Страница проекта после публикации GitHub Pages: https://turubayev.github.io/SteppeWatch-AI/

## Роль Claude

Подготовлена необязательная интеграция `scripts/claude_report.py`: рассчитанная числовая сводка → черновик объяснения на русском, казахском или английском → проверка специалистом. Claude не вычисляет спутниковые индексы, не подтверждает причины изменений и не принимает решения о землепользовании.

Рабочий путь API подготовлен, но реальный запрос не проверен: ключа и выделенных кредитов нет. По умолчанию запуск полностью локален и показывает запрос, а не выдуманный ответ Claude.

```bash
python scripts/claude_report.py results/kyzylorda.json
python scripts/claude_report.py results/kyzylorda.json --language Kazakh
```

Для реального запроса нужны локальные переменные `ANTHROPIC_API_KEY` и `ANTHROPIC_MODEL` с доступной в вашем Console моделью, затем явный флаг `--send`. Этот режим расходует API-кредиты; сейчас он не запускался.

```bash
python scripts/claude_report.py results/kyzylorda.json --language English --send
```

Передаются только разрешённые поля числовой сводки: годы, площадь, проценты классов, покрытие и ограничения метода. Растры, координаты, произвольные заметки и полевые записи не передаются. Ключ читается из окружения и не публикуется на сайте. Неполный SDG не превращается в полный показатель. Ответ модели всё равно требует проверки.

## English application draft

**Company / project name:** SteppeWatch-AI

**Short description:** Satellite land-monitoring tools for Kazakhstan, with human-reviewed explanations of screening results.

**What are you building?**

SteppeWatch-AI is an early-stage project developing satellite land-monitoring tools for Kazakhstan. Our research prototype combines vegetation time series and precipitation-adjusted trend analysis to identify areas for field investigation. It includes a Python analysis pipeline, a local Streamlit dashboard, synthetic tests and a saved satellite-data example from the Aral region. We target pasture managers and environmental organizations; customer traction and commercial demand have not yet been validated. Screening signals are not proof of degradation or land-use violations, and the prototype does not produce a certified UNCCD report.

**How will you use Claude?**

We plan to use the first-party Claude API to turn validated numerical summaries into concise Russian, Kazakh and English report drafts. We have prepared an optional API integration with an offline preview; live API calls have not yet been validated. Only allowlisted summary fields are included. Numerical geospatial analysis remains in our existing code. Claude should explain uncertainty and missing evidence, with every report reviewed by a specialist before delivery.

**Business model / next milestones:**

We intend to validate paid area-specific reports and recurring monitoring with potential users. Next milestones are independent field validation, user interviews, pilot reports and live evaluation of the Claude reporting integration. Pricing and revenue are not yet established.

## Перед подачей — заполнить реальные сведения в Console

- ФИО основателя, статус проекта/компании, страна и месяц/год начала.
- Зарегистрированное название и дата компании, только если компания действительно зарегистрирована.
- Финансирование, размер команды, клиенты и выручка — фактические значения, без предположений.
- Подтверждённый сайт и доступная почта на его домене, если форма их требует.

Для прежнего проекта была подана заявка на бесплатный домен `otvetik.eu.org`, № `20261008174805-arf-43778`. Это другое название; домен пока нельзя указывать как рабочий сайт SteppeWatch. Почта на нём не настроена. Не создавайте отдельные фиктивные компании или повторные заявки ради кредитов.

## Актуальные условия программы

На [официальной странице](https://claude.com/programs/startups) 9 октября 2026 года Anthropic сообщает о переполнении предложений с бесплатным годом Team и $1 000 API-кредитов. Их получение по новой заявке не обещано. Компания пересматривает заявки. Подача открыта для bootstrapped, pre-seed и венчурных проектов; остаются Startup Stack, встречи и мероприятия. Партнёрские предложения — не денежная выплата и могут иметь отдельные условия. Ориентир проверки заявки — до недели.

Заявка подаётся через официальный Claude Console по кнопке Apply now. Этот репозиторий не подтверждает соответствие условиям; окончательное решение принимает Anthropic. Заявка от имени основателя ещё не отправлена.

API: [Messages reference](https://platform.claude.com/docs/en/api/messages/create).
