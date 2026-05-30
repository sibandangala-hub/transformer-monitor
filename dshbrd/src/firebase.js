import { initializeApp } from 'firebase/app';
import { getDatabase } from 'firebase/database';

// 🔧 Replace with your Firebase project config
const firebaseConfig = {
  apiKey: "AIzaSyDUVDXompxEaHxq_XQa8dunQmR_-3IExcQ",
  authDomain: "transformer-db.firebaseapp.com",
  databaseURL: "https://transformer-db-default-rtdb.firebaseio.com",
  projectId: "transformer-db",
  storageBucket: "transformer-db.firebasestorage.app",
  messagingSenderId: "322028161520",
  appId: "1:322028161520:web:1887139ba1211e0cb93535"
};

const app = initializeApp(firebaseConfig);
export const db = getDatabase(app);
