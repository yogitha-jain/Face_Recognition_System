# Face Recognition System

A system that **enrolls people** and then **identifies new faces** by comparing them with the enrolled people. If the face does not match anyone well enough, it says **"unknown"**.

Built and tested on the LFW dataset with free, open-source models only. No paid APIs, no training.

## What it does

1. Finds faces in a photo (face detection).
2. Turns each face into a list of numbers (face embedding).
3. Compares it with every enrolled person using cosine similarity.
4. Picks the best match. If the score is below the threshold, the answer is **unknown**.

## Models used

| Model | Detector | Note |
|---|---|---|
| **ArcFace** (InsightFace) | SCRFD | Main model |
| FaceNet (facenet-pytorch) | MTCNN | For comparison |
| SFace (OpenCV) | YuNet | For comparison, and the fallback if InsightFace fails |

## Dataset

**LFW (Labeled Faces in the Wild)**: 13,233 photos of 5,749 people.
Download used: https://ndownloader.figshare.com/files/5976018 (checked with a SHA-256 checksum).
Original page: http://vis-www.cs.umass.edu/lfw/

## How it was tested

- **50 people** were enrolled, with 3 photos each.
- **Validation:** 2 more photos per person, used only to choose the threshold.
- **Test:** the remaining photos of those people (max 10 each), used once at the end. These photos are never the same as the enrollment photos.
- **Unknown people:** 4,848 people who were never enrolled, split into a validation half and a test half.
- **Threshold:** for each model, the lowest score at which at most **1%** of unknown validation people are accepted.

## Results (test set)

| Model | Threshold | Correct ID rate | False Reject Rate | False Accept Rate |
|---|---|---|---|---|
| ArcFace | 0.254 | 97.97% | 2.03% | 1.69% |
| FaceNet | 0.659 | 91.88% | 8.12% | 1.57% |
| SFace | 0.424 | 99.42% | 0.58% | 0.70% |

- **False Reject Rate:** a known person was rejected or matched to the wrong person.
- **False Accept Rate:** an unknown person was accepted as someone enrolled.
- No image was left without a detected face.
- All numbers come from `results/metrics.json`. Plots are in `results/`.

**Short summary:** all three models work well. SFace did best on this dataset. ArcFace is close, but accepted slightly more unknown people than the 1% target (1.69%). FaceNet rejected too many real people (8.12%).

<p align="center">
  <img src="results/arcface_scores.png" width="70%" alt="ArcFace scores">
</p>

## Where it fails

Ten example failures are saved in `results/failure_cases/`.

- **Photos with two people.** The system uses the largest face, which is sometimes the other person in the photo. For example, a photo of Rosalyn Carter with Jimmy Carter beside her.
- **People who appear together in photos** (like Barbara Boxer and Gray Davis) are sometimes confused.
- **Some people are hard for all three models** (for example Winona Ryder), probably because of the photos.
- Some LFW labels are noisy, which counts as a mistake even when the match is right.

## What could be improved

- Identify every face in a photo, not only the largest.
- Add a blur and pose check.
- Test with several random splits, not just one.
- Try photos that are not celebrity photos.

## How to run

1. Open `Face_Recognition_System.ipynb` in Google Colab.
2. Choose **Runtime, then Change runtime type, then T4 GPU**.
3. Click **Runtime, then Run all**. It takes about 10 to 20 minutes the first time.

It downloads the data and models, tests all three models, and saves the results in `results/`.

**Enroll and identify (example):**

```python
emb = get_embedder("arcface")
db = FaceDB(emb.name, emb.dim)

# enroll
face, status = select_face(emb.detect_embed(load_image(Path("alice.jpg"))))
db.add("Alice", face.embedding[None], model=emb.name, sources=["alice.jpg"])

# identify
probe, _ = select_face(emb.detect_embed(load_image(Path("new.jpg"))))
m = db.match(probe.embedding, threshold=0.254)
print(m.identity or "unknown", m.score)
```

## Files

- `Face_Recognition_System.ipynb`: the full code and evaluation
- `face_id_pipeline.py`: the pipeline code as a Python file
- `results/`: metrics, plots and failure examples
