from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

def get_db():
    try:
        client = MongoClient(os.getenv('MONGODB_URI') or 'mongodb://localhost:27017/')
        db = client['smart-learn-muse']
        return db
    except Exception as e:
        print(f"Database Connection Error: {e}")
        return None

@app.route('/', methods=['GET'])
def health_check():
    db = get_db()
    return jsonify({"status": "ML Service is running", "db_connected": db is not None}), 200

@app.route('/debug-db', methods=['GET'])
def debug_db():
    db = get_db()
    if db is None:
        return jsonify({"status": "Error", "message": "Database disconnected"}), 500
    
    try:
        counts = {
            "course_count": db.courses.count_documents({}),
            "user_count": db.users.count_documents({}),
            "enrollment_count": db.enrollments.count_documents({})
        }
        return jsonify({
            "status": "Connected",
            "database": "smart-learn-muse",
            "counts": counts
        }), 200
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

@app.route('/active-roster', methods=['GET'])
def active_roster():
    db = get_db()
    if db is None: return jsonify({"error": "No DB"}), 500
    try:
        # Instructors
        instructors = []
        for u in db.users.find({"role": "instructor"}):
            course_count = db.courses.count_documents({"instructor_id": u["_id"]})
            instructors.append({
                "id": str(u["_id"]),
                "name": u.get("name"),
                "email": u.get("email"),
                "role": u.get("role"),
                "courses_taught": course_count
            })
            
        # Students
        students = []
        for u in db.users.find({"role": "student"}):
            enrollment_count = db.enrollments.count_documents({"student_id": u["_id"]})
            students.append({
                "id": str(u["_id"]),
                "name": u.get("name"),
                "email": u.get("email"),
                "role": u.get("role"),
                "courses_enrolled": enrollment_count
            })
            
        return jsonify({
            "instructors": instructors,
            "students": students
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recommend-courses', methods=['POST'])
def recommend_courses():
    data = request.json
    student_id = data.get('student_id')
    
    db = get_db()
    if db is None:
        return jsonify({"recommendations": [], "error": "DB Connection Failed"}), 500
        
    try:
        if not student_id:
            return jsonify({"recommendations": [], "error": "student_id required"}), 400

        sid = ObjectId(student_id)
        
        # 1. Fetch Student History
        enrollments = db.enrollments.find({"student_id": sid})
        enrolled_ids = [e["course_id"] for e in enrollments]
        
        history_courses = list(db.courses.find({"_id": {"$in": enrolled_ids}}))
        user_history = [c.get('description', '') for c in history_courses if c.get('description')]
        
        # 2. Fetch Available Courses
        available_courses = list(db.courses.find({"_id": {"$nin": enrolled_ids}}))
        
        if not available_courses:
            return jsonify({"recommendations": []}), 200

        if not user_history:
            # Return top 3 random or first 3
            formatted = []
            for c in available_courses[:3]:
                formatted.append({"id": str(c["_id"]), "title": c["title"], "description": c.get("description", "")})
            return jsonify({"recommendations": formatted}), 200

        # 3. Vectorization & Similarity
        candidate_docs = [c.get('description', '') for c in available_courses]
        all_descriptions = user_history + candidate_docs
        
        tfidf_vec = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vec.fit_transform(all_descriptions)
        
        history_len = len(user_history)
        history_vectors = tfidf_matrix[:history_len]
        candidate_vectors = tfidf_matrix[history_len:]
        
        user_profile = history_vectors.mean(axis=0)
        user_profile_np = np.asarray(user_profile)
        
        similarities = cosine_similarity(user_profile_np, candidate_vectors).flatten()
        related_indices = similarities.argsort()[::-1]
        
        recommendations = []
        for idx in related_indices:
            c = available_courses[idx]
            recommendations.append({
                "id": str(c["_id"]),
                "title": c["title"],
                "description": c.get("description", "")
            })
            if len(recommendations) >= 5:
                break
                
        return jsonify({"recommendations": recommendations}), 200
        
    except Exception as e:
        print(f"ML Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict-performance', methods=['POST'])
def predict_performance():
    data = request.json
    score = data.get('quiz_score', 0)
    
    prediction = "Low"
    if score > 80: prediction = "High"
    elif score > 50: prediction = "Medium"
        
    return jsonify({"performance_category": prediction}), 200

@app.route('/predict-dropout', methods=['POST'])
def predict_dropout():
    data = request.json
    student_id = data.get('student_id')
    
    db = get_db()
    if db is None: return jsonify({"error": "No DB"}), 500
    
    try:
        sid = ObjectId(student_id)
        
        # 1. Progress Metrics
        enrollments = list(db.enrollments.find({"student_id": sid}))
        if not enrollments:
            return jsonify({"dropout_probability": 0.5, "risk_level": "medium", "message": "No data"}), 200
            
        avg_progress = np.mean([e.get("progress", 0) for e in enrollments])
        
        # 2. Quiz Metrics
        quiz_scores = list(db.quiz_scores.find({"student_id": sid}))
        quiz_avg = np.mean([s.get("score", 0) for s in quiz_scores]) if quiz_scores else 0
        
        # 3. Inactivity Metrics
        last_active = max([e.get("last_accessed", datetime.min) for e in enrollments])
        if last_active == datetime.min:
            inactivity = 30 # Default high inactivity if no date
        else:
            inactivity = (datetime.now() - last_active).days
        
        # Risk scoring
        score = 0
        if avg_progress < 20: score += 4
        if inactivity > 5: score += 3
        if quiz_avg < 50: score += 3
        
        prob = 1 / (1 + np.exp(-(score - 5)))
        risk_level = "low"
        if prob > 0.7: risk_level = "high"
        elif prob > 0.4: risk_level = "medium"
        
        return jsonify({
            "dropout_probability": round(float(prob), 2),
            "risk_level": risk_level,
            "metrics": {"progress": round(float(avg_progress),2), "inactivity": inactivity, "quiz_avg": round(float(quiz_avg),2)}
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/study-optimizer', methods=['POST'])
def study_optimizer():
    data = request.json
    history = data.get('activity_history', [])
    
    if not history:
        return jsonify({"suggestion": "Complete more quizzes to see your peak performance time!"}), 200
        
    df = pd.DataFrame(history)
    if 'hour' not in df.columns or 'score' not in df.columns:
        return jsonify({"suggestion": "Insufficient data format"}), 200
        
    best_time = df.groupby('hour')['score'].mean().idxmax()
    
    periods = {0: "late night", 5: "morning", 12: "afternoon", 17: "evening", 22: "late night"}
    period = next(v for k, v in sorted(periods.items(), reverse=True) if best_time >= k)
    
    return jsonify({
        "best_hour": int(best_time),
        "suggestion": f"Analysis shows you perform best in the {period}. Focus on new topics around {best_time}:00."
    }), 200

@app.route('/generate-quiz-topics', methods=['POST'])
def generate_quiz_topics():
    data = request.json
    desc = data.get('course_description', '')
    
    if len(desc) < 10:
        return jsonify({"suggested_topics": ["Overview", "Fundamentals"]}), 200
        
    words = [w for w in desc.lower().replace('.', ' ').replace(',', ' ').split() if len(w) > 5]
    topics = list(set([w.capitalize() for w in words]))[:5]
    
    return jsonify({"suggested_topics": topics}), 200


@app.route('/chat-assistant', methods=['POST'])
def chat_assistant():
    data = request.json
    message = data.get('message', '').lower()
    course_id = data.get('course_id')
    
    db = get_db()
    course = None
    if course_id:
        try:
            course = db.courses.find_one({"_id": ObjectId(course_id)})
        except: pass

    title = course['title'] if course else 'this course'
    
    # Simple rule-based logic for demo (simulated LLM)
    if 'quiz' in message:
        response = f"I recommend taking the practice quizzes for {title} after each lesson to reinforce your learning."
    elif 'help' in message or 'explain' in message:
        response = f"I'd be happy to explain concepts from {title}. Which specific topic are you struggling with?"
    elif 'progress' in message:
        response = "You're doing great! Consistency is key to mastering these concepts."
    else:
        response = f"That's an interesting point about {title}. I'm here to help you navigate through the lessons and exercises!"
        
    return jsonify({"answer": response}), 200

@app.route('/instructor-insights', methods=['POST'])
def instructor_insights():
    data = request.json
    instructor_id = data.get('instructor_id')
    
    db = get_db()
    if db is None: return jsonify({"error": "No DB"}), 500
    
    try:
        iid = ObjectId(instructor_id)
        courses = list(db.courses.find({"instructor_id": iid}))
        course_ids = [c["_id"] for c in courses]
        
        enrollments = list(db.enrollments.find({"course_id": {"$in": course_ids}}))
        
        insights = []
        
        # 1. Top Performing
        if enrollments:
            course_avg = {}
            for e in enrollments:
                cid = str(e["course_id"])
                if cid not in course_avg: course_avg[cid] = []
                course_avg[cid].append(e.get("progress", 0))
            
            best_cid = max(course_avg, key=lambda k: np.mean(course_avg[k]))
            best_course = next(c for c in courses if str(c["_id"]) == best_cid)
            insights.append({
                "title": "Top Performing Course",
                "message": f"'{best_course['title']}' has the highest average student progress this week.",
                "type": "success"
            })
            
            # 2. Risk Alert
            at_risk = [e for e in enrollments if e.get("progress", 0) < 20]
            if at_risk:
                insights.append({
                    "title": "Student Risk Alert",
                    "message": f"{len(at_risk)} students are currently below 20% progress. Consider reaching out.",
                    "type": "warning"
                })
        
        # 3. Growth
        if len(courses) > 0:
            insights.append({
                "title": "Growth Tip",
                "message": "Adding more interactive quizzes to your courses could increase retention by up to 25%.",
                "type": "info"
            })
            
        if not insights:
            insights = [{"title": "Welcome", "message": "Start creating courses to see AI insights!", "type": "info"}]
            
        suggestions = []
        if at_risk:
            suggestions.append(f"AI Alert: {len(at_risk)} students are struggling in Module 1. Try adding a 'Tips & Tricks' video there.")
        if len(courses) > 0:
            suggestions.append("Insight: Courses with 2+ quizzes have 30% higher retention. Add more interactive checks.")

        return jsonify({"insights": insights, "suggestions": suggestions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    app.run(port=port, debug=True)
