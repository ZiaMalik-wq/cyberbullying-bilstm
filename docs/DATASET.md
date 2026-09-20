# Dataset Documentation

## Source

Kaggle: "Cyberbullying Classification" dataset
(originally sourced from Twitter/X social-media posts)

Kaggle URL: <https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification>

## Dataset File

| Property | Value |
| ---------- | ------- |
| Filename | `cyberbullying_tweets.csv` |
| Location | `data/raw/cyberbullying_tweets.csv` |
| Format | CSV |
| Encoding | UTF-8 |
| Size | 7,174,545 bytes (6.84 MB) |

## Number of Samples

**47,692 rows** (including header)

## Columns

| # | Column Name           | Dtype | Description        |
|---|-----------------------|-------|--------------------|
| 1 | `tweet_text`          | str   | Raw tweet text     |
| 2 | `cyberbullying_type`  | str   | Target class label |

## Text Column

`tweet_text`

## Target Column

`cyberbullying_type`

## Classes

Six classes are present in the dataset. The **exact label strings** as they
appear in the file are:

| Label String | Count | Percentage |
| -------------- | ------: | ----------: |
| `religion` | 7,998 | 16.77% |
| `age` | 7,992 | 16.76% |
| `gender` | 7,973 | 16.72% |
| `ethnicity` | 7,961 | 16.69% |
| `not_cyberbullying` | 7,945 | 16.66% |
| `other_cyberbullying` | 7,823 | 16.40% |
| **Total** | **47,692** | **100%** |

The dataset is **nearly perfectly balanced** across all six classes.
The maximum imbalance is `religion` (7,998) vs `other_cyberbullying` (7,823):
a difference of 175 samples (0.37 percentage points).

### Alignment with PROJECT_SPEC.md

The six spec categories map to the dataset labels as follows:

| PROJECT_SPEC.md Category | Dataset Label |
| -------------------------- | --------------- |
| Age | `age` |
| Ethnicity | `ethnicity` |
| Gender | `gender` |
| Religion | `religion` |
| Other Cyberbullying | `other_cyberbullying` |
| Not Cyberbullying | `not_cyberbullying` |

A canonical integer mapping will be defined in `src/config.py` during the
implementation phase. The exact mapping has not been defined yet.

## Missing Values

| Column | Missing Count | Missing % |
| -------- | -------------: | ----------: |
| `tweet_text` | 0 | 0.0% |
| `cyberbullying_type` | 0 | 0.0% |

No missing values in either column.

## Duplicate Rows

| Type | Count |
| ------ | ------: |
| Exact duplicate rows (identical text AND label) | 36 |
| Duplicate `tweet_text` rows (same text, any label) | 1,675 |
| Texts with conflicting labels (same text, different labels) | 1,639 |

> [!WARNING]
> **1,639 tweets appear with more than one label.**
> This is a significant data-quality issue. The same tweet text has been
> assigned to different cyberbullying categories in different rows.
> This must be addressed during the preprocessing phase (after the split)
> to avoid a tweet from the training set leaking into the test set under
> a different label.

## Character-Length Statistics (`tweet_text`)

| Statistic | Characters |
| ----------- | ----------: |
| Minimum | 1 |
| Maximum | 5,018 |
| Mean | 136.25 |
| Median | 124.0 |
| Std Dev | 85.23 |
| 25th percentile | 78.0 |
| 75th percentile | 180.0 |
| 90th percentile | 269.0 |
| 95th percentile | 277.0 |
| 99th percentile | 284.0 |

Note: The maximum of 5,018 characters is an extreme outlier. The 95th
percentile is 277 characters, consistent with the old 280-character Twitter
limit. Some samples exceed this limit considerably, possibly due to
concatenated text or data collection artefacts.

## Word/Token-Length Statistics (`tweet_text`, whitespace split)

| Statistic | Tokens |
| ----------- | -------: |
| Minimum | 1 |
| Maximum | 790 |
| Mean | 23.70 |
| Median | 20.0 |
| Std Dev | 15.43 |
| 25th percentile | 13.0 |
| 75th percentile | 32.0 |
| 90th percentile | 47.0 |
| 95th percentile | 51.0 |
| 99th percentile | 56.0 |

The maximum of 790 tokens is an extreme outlier. The 99th percentile is 56
tokens. A sequence-length cap of 50–100 tokens will cover the vast majority
of samples. Exact value to be decided through hyperparameter experiments.

## Social-Media Feature Presence

| Feature | Count | Percentage |
| --------- | ------: | ----------: |
| URLs (`http://` / `https://` / `www.`) | 4,313 | 9.0% |
| @mentions | 18,098 | 37.9% |
| #hashtags | 6,964 | 14.6% |
| Emoji characters | 427 | 0.9% |
| Repeated characters (3+ consecutive) | 7,151 | 15.0% |

> [!NOTE]
> @mentions (37.9%) and repeated characters (15%) are the most prevalent
> social-media features. Whether to preserve, normalize, or remove these
> must be decided explicitly during the preprocessing design phase — not
> automatically. Per PROJECT_SPEC.md §9, aggressive removal of linguistic
> information is discouraged.

## Non-ASCII Characters

| Metric                                     |  Value |
|--------------------------------------------|-------:|
| Rows with at least one non-ASCII character |  9,161 |
| Percentage                                 |  19.2% |

Non-ASCII content includes Arabic, Turkish, Portuguese, and other
non-English scripts, as well as emoji. The file is valid UTF-8.

Example rows with non-ASCII content:

```text
RT @Kurdsnews: Turkish state has killed 241 children in last 11 years
http://t.co/JlvkE1epws #news ##GoogleÇeviriciTopluluğuKürtçeyideE…

Kids Love😘❤ @ Mohamad Bin Zayed City مدينة محمد بن زايد http://t.co/0xrOZSNn

@sand_dejesus Isso é bullying! @O_Patriarca
```

## Representative Examples Per Class

### `age`

```text
Here at home. Neighbors pick on my family and I. Mind you my son is autistic.
It feels like high school. They call us names attack us for no reason and bully
us all the time. Can't step on my front po[rch]...

Being bullied at school: High-achieving boys use further strategies to maintain
masculinity, for example becoming bullies themselves, disrupting the lessons, or
devaluing girls' achievements https://w[...]
```

### `ethnicity`

```text
Hey dumb fuck celebs stop doing something for people for publicity on Facebook...
Wtf happen to life u niggers are cowards.

Fuck u bitch RT @tayyoung_: FUCK OBAMA, dumb ass nigger
```

### `gender`

```text
rape is real..zvasiyana nema jokes about being drunk or being gay or being
lesbian...rape is not ones choice or wish..thtz where the sensitivity is coming
from

You never saw any celebrity say anything like this for Obama: B Maher Incest
Rape 'Joke' S Colbert Gay 'joke' K Griffin beheading 'joke'
```

### `not_cyberbullying`

```text
In other words #katandandre, your food was crapilicious! #mkr

Why is #aussietv so white? #MKR #theblock #ImACelebrityAU #today #sunrise
#studio10 #Neighbours #WonderlandTen #etc
```

### `other_cyberbullying`

```text
@ikralla fyi, it looks like I was caught by it. I'm not a botter, so...

I need to just switch to an organization-based github, but I don't want to pay
$25/month because I'm cheap. :\
```

### `religion`

```text
Sudeep, did she invite him though? No right? Why are you getting worded up?
You're okay with Parvesh Verma cause he speaks against Muslims but against an
idiot like Imam because he called for chakka j[am]...

@discerningmumin Islam has never been a resistance to oppression. It has always
been source of oppression to both believers and non believer
```

## Data Quality Issues

| Issue | Detail | Severity |
| ------- | -------- | ---------- |
| Same tweet with conflicting labels | 1,639 tweets assigned to more than one class | **High** |
| Exact duplicate rows | 36 rows are fully identical | Medium |
| Extreme outlier lengths | Max 5,018 chars / 790 words; p99 is 284 chars / 56 words | Medium |
| Non-English content | 19.2% of rows contain non-ASCII; some are fully non-English | Low–Medium |
| `not_cyberbullying` category quality | Example rows appear genuinely non-bullying but sometimes contain borderline content (see `other_cyberbullying` examples which appear innocuous) | Medium |
| Very short texts | Minimum length is 1 character; some very short samples may be uninformative | Low |

## Notes

- All statistics above are verified from the actual file by running
  `src/inspect_dataset.py` on 2026-09-21.
- The raw file has **not been modified**.
- A machine-readable JSON summary is stored in `results/dataset_inspection.json`.
- The conflicting-label issue (1,639 samples) is the most significant
  data-quality concern and must be addressed during the preprocessing design
  phase before any train/val/test split is performed.
