# Dataset

[Google Speech Commands](https://arxiv.org/abs/1804.03209) (Warden, 2018) is a crowd-sourced set of about 65,000 one-second clips covering 35+ short words. Each clip was recorded by prompting a volunteer to say a single word. It is released under CC BY 4.0.

## Commands and data

- **Commands used:** follow, stop, go, learn, off.
- **Clips per command:** each command has roughly 1,500 to 4,000 clips. 1,300 pseudo-random clips per word were used for training.
- **Negative examples:** all other words, about 20,000 samples from 10 non-command words, are treated as "not the command" by each binary detector.
- **Problem clips:** clips shorter than one second, which can't produce a full 16 × 16 MFCC matrix, are dropped. These are mostly recordings that started or stopped too early.

## Speaker token density (paper, Figs. 3–5)

Speaker metadata is limited, but each clip carries an anonymous speaker hash. The paper looked at how many clips each speaker contributed:

- For the short commands (*go, stop, off*), about 43–45% of speakers contributed a single clip.
- For *follow* and *learn*, more speakers contributed four or five clips.

The longer, higher-perplexity commands (*follow, learn*) generally scored lower. The spread of speakers argues against overfitting as the reason the short commands did well.

## An earlier in-house collection

A phone-app collection of custom wake phrases was gathered with UTD computer-science students. It was dropped in favour of Speech Commands because the clip lengths were inconsistent. That audio is not public and is not part of this repository.
