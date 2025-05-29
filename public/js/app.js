// Import the functions you need from the SDKs you need
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getAnalytics } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-analytics.js";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyAFG8OO7p6Nzx3PtLILjytRG4FSFNSxddw",
  authDomain: "eastern-perigee-461323-s4.firebaseapp.com",
  projectId: "eastern-perigee-461323-s4",
  storageBucket: "eastern-perigee-461323-s4.firebasestorage.app",
  messagingSenderId: "375807967146",
  appId: "1:375807967146:web:8e61b1f27abf2879d2b5d3",
  measurementId: "G-R7ZKZBZJ8T"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
