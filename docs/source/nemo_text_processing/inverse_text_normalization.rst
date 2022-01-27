Inverse Text Normalization
==========================


Inverse text normalization (ITN) is a part of the Automatic Speech Recognition (ASR) post-processing pipeline.
ITN is the task of converting the raw spoken output of the ASR model into its written form to improve text readability.

For example,
`"in nineteen seventy"` -> `"in 1975"`
and `"it costs one hundred and twenty three dollars"` -> `"it costs $123"`.

Test
:cite:`itn-kurzinger2022ctc`


References
----------

.. bibliography:: textprocessing_all.bib
    :style: plain
    :labelprefix: ITN
    :keyprefix: itn-