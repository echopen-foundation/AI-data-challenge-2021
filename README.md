# AI-data-challenge-2021

## Data challenge background

Pulmonary ultrasound may be an alternative tool to screen patients with or suspected of having COVID-19. A prospective study carried out at the AP-HP evaluated the correlation between a pulmonary ultrasound severity score and the unfavorable clinical evolution of the patient under 28 days.

In this data challenge, you will need to use a machine Learning approach on data from this study to construct a global predictor of the patient's clinical outcome.

More details can be found [HERE](ProjectDescription.md)

## How to post a submission

At the end of the hackathon, you will be required to share your methods and source code with the open community under [CC by NC 2.0 license](https://creativecommons.org/licenses/by-nc/2.0/)

We suggest the following **submission template**:

#### Project name

ACUPen

#### Project description

- Data wrangling
- Data augmentation
- Training using XGBoost

#### Team members

* Marc DEMOUSTIER
* Quoc Duong NGUYEN

#### Preprocessing and training methods

Explain in details the preprocessing and training methods(dataset split and stratification, hyperparameters selection) you are using to build your global predictor of the patient's clinical outcome.
You will also provide a link to the source code implementing these different steps.

#### Model performances
The model performances are evaluated on a hidden subset of the dataset following the [F1-score](https://towardsdatascience.com/the-f1-score-bec2bbc38aa6) applied to patient's clinical outcome variable.

#### Medical discussion and model interpretability

In this section, you will provide meaningfull visualizations and interpretations that will help to understand the underlying dynamic of the model decision. It will nurture a medical discussion with the radiologists involved in the challenge.  

## Challenge partners

* Epita
* echOpen foundation
* AP-HP
* EIT Health
