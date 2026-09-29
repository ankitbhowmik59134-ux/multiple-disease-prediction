# Multiple Disease Prediction System

Java Spring Boot app that trains three logistic-regression models at startup and serves predictions from a web UI and a JSON API.

Models:

- **Diabetes** — glucose, BMI, age, and related markers
- **Heart disease** — age, cholesterol, blood pressure, exercise findings
- **Parkinson’s** — voice biomarkers (jitter, shimmer, HNR, PPE)

This is an educational demo. It is **not** a medical diagnosis tool.

## Run

Java 17+ is required (Java 24 works). Maven Wrapper is included.

**Windows (PowerShell or Command Prompt):**

```bat
mvnw.cmd test
mvnw.cmd spring-boot:run
```

**macOS / Linux:**

```bash
chmod +x mvnw
./mvnw test
./mvnw spring-boot:run
```

Then open [http://localhost:8080](http://localhost:8080).

## Deploy

This is a **Java Spring Boot** process, so it cannot run on Vercel (Vercel hosts static sites and short-lived serverless functions, not a JVM web server).

Use a container host instead. The repo includes a `Dockerfile`.

### Render (recommended)

1. Open [https://render.com/deploy?repo=https://github.com/ankitbhowmik59134-ux/multiple-disease-prediction](https://render.com/deploy?repo=https://github.com/ankitbhowmik59134-ux/multiple-disease-prediction)
2. Sign in with GitHub and create the web service.
3. Wait for the Docker build, then open the `onrender.com` URL.

Local container check:

```bash
docker build -t mdps .
docker run --rm -p 8080:8080 -e PORT=8080 mdps
```

## Verify

1. **Tests:** `mvnw.cmd test` (Windows) or `./mvnw test` (macOS/Linux) — logistic regression should separate a simple dataset; the API should list 3 diseases and accept a diabetes payload.
2. **Home page:** three cards with hold-out accuracy (typically ~80%+ on the generated training data).
3. **UI flow:** open Diabetes → submit the prefilled values → you should see a probability and a Low / Moderate / High risk label.
4. **API:**

```bash
curl http://localhost:8080/api/v1/diseases
```

```bash
curl -X POST http://localhost:8080/api/v1/predict/heart -H "Content-Type: application/json" -d "{\"age\":58,\"sex\":1,\"chestPainType\":3,\"restingBp\":150,\"cholesterol\":310,\"fastingBloodSugar\":1,\"maxHeartRate\":112,\"exerciseAngina\":1,\"oldpeak\":2.8,\"numVessels\":2}"
```

A high-risk heart profile should return `"positive": true` and a probability above 0.5.

## How the ML works

On startup, each disease generator builds 900 labeled samples from a noisy clinical rule. Features are standardized, then binary logistic regression is trained with L2-regularized gradient descent. Accuracy is reported on an 80/20 hold-out split.
