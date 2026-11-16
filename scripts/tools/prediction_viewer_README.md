# MT prediction viewer

Build the offline page from the repository root (no dependencies):

```powershell
& C:\ProgramData\anaconda3\python.exe scripts\tools\build_prediction_viewer.py
```

Open `prediction_review.html` beside the original prediction CSV in your browser.
The page embeds the data and works without a server or internet connection.
For another run, pass its CSV path, optionally followed by `--output PATH`.

Features: direction and review filters, text/ID search, ascending metric sorting,
pagination, jump to example ID, source/prediction/reference comparison, optional
literal word diff, ratings (1–5), notes and follow-up flags. Arrow keys navigate;
1–5 rate, F flags, and N selects the next unreviewed item outside form controls.
The built-in review guide explains rating anchors and metric limitations.

Reviews auto-save to browser local storage, keyed by the original CSV SHA-256.
Browser storage may be unavailable for local files, private browsing, or disabled
storage; the page warns when saving fails. Keep regular **Export JSON** backups.
**Import reviews** merges JSON from the exact same dataset; imported entries
replace existing entries with the same ID. **Export CSV** includes every original
row and metric plus human ratings, flags, notes and timestamps. A note or flag
also counts as a reviewed example; the mean human rating uses rated examples only.
Filter membership and ordering refresh when filters change; editing the current
review keeps it on screen so you can finish notes before moving on.

The generated page and review exports are run artifacts, not changes to the
prediction CSV. No records are uploaded. Local-file storage behavior and visual
layout should be checked in your preferred browser.
