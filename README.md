# Leaf Disease Detection Multiclass Frontend

This is the separate frontend for the multiclass models. It is isolated from the existing binary frontend in `frontend_P`.

Routes:

- `/` for the landing page
- `/detect` for the multiclass detection workspace with a model selector

## Run locally

1. Install dependencies

```bash
npm install
```

2. Start the multiclass frontend

```bash
npm run dev
```

3. Start the multiclass backend from the project root in another terminal

```bash
python app_multiclass.py
```

The Vite dev server runs on `http://127.0.0.1:5174` and proxies `/predict` and `/chat` to `http://127.0.0.1:5001`.

The backend supports both:

- `InceptionV3`
- `ResNet50`

## Optional environment variable

Create `Frontend_Multi/.env` if you want the frontend to call a custom backend URL directly:

```bash
VITE_API_BASE_URL=http://127.0.0.1:5001
```
