# Leaf Disease Detection Frontend

Frontend rebuilt with a clearer folder structure and a two-step flow:

- Landing page at `/`
- Detection workspace at `/detect`

## Structure

```text
src/
  app/
  components/
    chat/
    detection/
    landing/
    layout/
  constants/
  hooks/
  pages/
  services/
  styles/
```

## Run locally

1. Install dependencies

```bash
npm install
```

2. Start the frontend

```bash
npm run dev
```

3. Start the backend from the project root in another terminal

```bash
python app.py
```

The Vite dev server proxies `/predict` and `/chat` to `http://127.0.0.1:5000`.

## Optional environment variable

Create `frontend_P/.env` if you want the frontend to call a custom backend URL directly:

```bash
VITE_API_BASE_URL=http://127.0.0.1:5000
```
