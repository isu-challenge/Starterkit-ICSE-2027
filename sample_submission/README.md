# Sample submission

Templates for the folder you encrypt and submit. The rules, the runtime environment, and
how to test locally are described under "Official Submission" in the
[Track A README](../track_a/README.md#official-submission) and the
[Track B README](../track_b/README.md#official-submission).

```text
track_a/
├── translator.py           template translator with create_translator(device)
├── models.txt              Hugging Face models your translator loads
└── requirements.txt        extra PyPI packages
track_b/
├── system.py               template ISU system with create_system(device)
├── models.txt              Hugging Face models your system loads
└── requirements.txt        extra PyPI packages
```

Copy a template into your local submission folder, for example
`mkdir -p track_b/submissions && cp -r sample_submission/track_b track_b/submissions/<team_name>`.
