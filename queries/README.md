Drop the organisers' `query-*.txt` files here.

The task is taken from the filename suffix, exactly as the rules specify:

| suffix   | task        | output columns                     |
|----------|-------------|------------------------------------|
| `-kis`   | Textual KIS | `video_id,frame_id`                |
| `-qa`    | Q&A         | `video_id,frame_id,answer`         |
| `-trake` | TRAKE       | `video_id,frame_1,...,frame_N`     |

Each result CSV keeps the query's own stem: `query-1-kis.txt` → `query-1-kis.csv`.
The three files here are the examples from the rulebook, kept as a smoke test.
