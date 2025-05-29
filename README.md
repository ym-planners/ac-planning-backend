# Firebase Web Hosting Setup

This project is configured for Firebase Hosting.

## Prerequisites

- Node.js and npm installed (for Firebase CLI)
- Firebase account

## Deployment Steps

1.  **Install Firebase CLI (if not already installed):**
    ```bash
    npm install -g firebase-tools
    ```

2.  **Login to Firebase:**
    ```bash
    firebase login
    ```

3.  **Initialize Firebase in your project (only if you haven't cloned this repo with .firebaserc and firebase.json):**
    If you cloned this repository, the necessary Firebase configuration files (`.firebaserc` and `firebase.json`) are already included. You might be asked to associate the project with your Firebase account if it's the first time.
    If you are setting this up in a new local directory for an existing Firebase project, you might run:
    ```bash
    firebase init hosting
    ```
    And select your existing project ("eastern-perigee-461323-s4"). Make sure to choose "public" as the public directory.

4.  **Deploy to Firebase Hosting:**
    ```bash
    firebase deploy --only hosting
    ```

5.  **Access your site:**
    After deployment, the Firebase CLI will provide you with the URL to your hosted site (e.g., `https://<YOUR_PROJECT_ID>.web.app`).

---

# ac-planning-backend
ac-planning-backend
