from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import bcrypt
import jwt
from emergentintegrations.llm.chat import LlmChat, UserMessage
import asyncio
import math

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Environment variables
JWT_SECRET = os.environ.get('JWT_SECRET', 'fallback-secret')
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY', '')

# Create app
app = FastAPI()
api_router = APIRouter(prefix="/api")
security = HTTPBearer()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ MODELS ============

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    name: str
    city: Optional[str] = None
    province: Optional[str] = None
    radius: int = 25  # km
    job_types: List[str] = []
    skills: List[str] = []
    cv_files: List[Dict[str, str]] = []  # [{"id": "...", "name": "...", "url": "..."}]
    is_admin: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class Subscription(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    plan: str  # "Free", "Starter", "Premium", "VIP"
    credits_remaining: int
    credits_monthly: int
    status: str = "active"  # active, cancelled, expired
    next_billing_date: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class Opportunity(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    category: str
    source: str  # "CareerJunction", "Indeed", "Gumtree", etc.
    apply_url: str
    posted_at: str
    location: str
    location_lat: Optional[float] = None
    location_lng: Optional[float] = None
    remote_flag: bool = False
    skills_required: List[str] = []

class Application(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    opportunity_id: str
    status: str = "Queued"  # Queued, Submitted, Viewed, Replied, Interview, Offer, Rejected
    cv_used: Optional[str] = None
    cover_letter: Optional[str] = None
    ai_prompt_used: Optional[str] = None
    match_score: Optional[int] = None
    requires_user_action: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class Wallet(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    balance: float = 0.0
    transactions: List[Dict[str, Any]] = []  # [{"type": "credit", "amount": 50, "reason": "referral", "date": "..."}]

class Referral(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    referrer_user_id: str
    referred_user_id: Optional[str] = None
    referred_email: Optional[str] = None
    status: str = "pending"  # pending, completed
    referral_code: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

# ============ REQUEST/RESPONSE MODELS ============

class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    name: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class AuthResponse(BaseModel):
    token: str
    user: User

class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    province: Optional[str] = None
    radius: Optional[int] = None
    job_types: Optional[List[str]] = None
    skills: Optional[List[str]] = None

class AutoApplyRequest(BaseModel):
    opportunity_id: str
    cv_id: Optional[str] = None

class UpgradePlanRequest(BaseModel):
    plan: str

# ============ AUTH HELPERS ============

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(days=30)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload["user_id"]
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ============ MATCH SCORE ALGORITHM ============

def calculate_match_score(user: User, opportunity: Opportunity) -> int:
    """Calculate match score 0-100 based on skills, location, job type"""
    score = 0
    
    # Skills match (50 points)
    user_skills_lower = [s.lower() for s in user.skills]
    opp_skills_lower = [s.lower() for s in opportunity.skills_required]
    if opp_skills_lower:
        matching_skills = sum(1 for s in opp_skills_lower if s in user_skills_lower)
        score += int((matching_skills / len(opp_skills_lower)) * 50)
    else:
        score += 25  # No specific skills required
    
    # Location match (30 points)
    if opportunity.remote_flag:
        score += 30
    elif user.city and opportunity.location:
        if user.city.lower() in opportunity.location.lower():
            score += 30
        else:
            score += 10  # Different location
    
    # Job type match (20 points)
    if user.job_types:
        if any(jt.lower() in opportunity.category.lower() for jt in user.job_types):
            score += 20
    else:
        score += 10
    
    return min(score, 100)

# ============ AUTH ROUTES ============

@api_router.post("/auth/signup", response_model=AuthResponse)
async def signup(req: SignupRequest):
    # Check if user exists
    existing = await db.users.find_one({"email": req.email}, {"_id": 0})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user = User(
        email=req.email,
        name=req.name,
        is_admin=(req.email in ["admin@aimoney.sa", "admin@example.com"])  # Admin emails
    )
    user_dict = user.model_dump()
    user_dict["password"] = hash_password(req.password)
    await db.users.insert_one(user_dict)
    
    # Create subscription (Free plan)
    subscription = Subscription(
        user_id=user.id,
        plan="Free",
        credits_remaining=3,
        credits_monthly=3
    )
    await db.subscriptions.insert_one(subscription.model_dump())
    
    # Create wallet
    wallet = Wallet(user_id=user.id)
    await db.wallets.insert_one(wallet.model_dump())
    
    # Create token
    token = create_token(user.id)
    
    return {"token": token, "user": user}

@api_router.post("/auth/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    user_doc = await db.users.find_one({"email": req.email}, {"_id": 0})
    if not user_doc:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not verify_password(req.password, user_doc.get("password", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user = User(**user_doc)
    token = create_token(user.id)
    
    return {"token": token, "user": user}

# ============ USER ROUTES ============

@api_router.get("/user/profile", response_model=User)
async def get_profile(user_id: str = Depends(get_current_user)):
    user_doc = await db.users.find_one({"id": user_id}, {"_id": 0})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    return User(**user_doc)

@api_router.put("/user/profile", response_model=User)
async def update_profile(req: UpdateProfileRequest, user_id: str = Depends(get_current_user)):
    update_data = {k: v for k, v in req.model_dump().items() if v is not None}
    if update_data:
        await db.users.update_one({"id": user_id}, {"$set": update_data})
    
    user_doc = await db.users.find_one({"id": user_id}, {"_id": 0})
    return User(**user_doc)

@api_router.post("/user/cv/upload")
async def upload_cv(file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    # Mock upload (save filename only)
    cv_id = str(uuid.uuid4())
    cv_entry = {
        "id": cv_id,
        "name": file.filename,
        "url": f"/uploads/{cv_id}_{file.filename}",
        "uploaded_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.users.update_one(
        {"id": user_id},
        {"$push": {"cv_files": cv_entry}}
    )
    
    return {"success": True, "cv": cv_entry}

# ============ OPPORTUNITIES ROUTES ============

@api_router.get("/opportunities", response_model=List[Dict[str, Any]])
async def get_opportunities(user_id: str = Depends(get_current_user)):
    # Get user profile
    user_doc = await db.users.find_one({"id": user_id}, {"_id": 0})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    user = User(**user_doc)
    
    # Get all opportunities
    opportunities = await db.opportunities.find({}, {"_id": 0}).to_list(1000)
    
    # Calculate match scores
    results = []
    for opp_doc in opportunities:
        opp = Opportunity(**opp_doc)
        match_score = calculate_match_score(user, opp)
        results.append({
            **opp.model_dump(),
            "match_score": match_score
        })
    
    # Sort by match score descending
    results.sort(key=lambda x: x["match_score"], reverse=True)
    
    return results

# ============ APPLICATIONS ROUTES ============

@api_router.post("/applications/auto-apply")
async def auto_apply(req: AutoApplyRequest, user_id: str = Depends(get_current_user)):
    # Check subscription credits
    sub_doc = await db.subscriptions.find_one({"user_id": user_id}, {"_id": 0})
    if not sub_doc:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    sub = Subscription(**sub_doc)
    
    # Admin has unlimited credits
    user_doc = await db.users.find_one({"id": user_id}, {"_id": 0})
    is_admin = user_doc.get("is_admin", False)
    
    if not is_admin and sub.credits_remaining <= 0:
        raise HTTPException(status_code=403, detail="Insufficient credits. Please upgrade your plan.")
    
    # Get opportunity
    opp_doc = await db.opportunities.find_one({"id": req.opportunity_id}, {"_id": 0})
    if not opp_doc:
        raise HTTPException(status_code=404, detail="Opportunity not found")
    opp = Opportunity(**opp_doc)
    
    # Get user
    user = User(**user_doc)
    
    # Calculate match score
    match_score = calculate_match_score(user, opp)
    
    # Generate cover letter with AI
    cover_letter = ""
    ai_prompt = f"""Generate a professional cover letter for this job:

Job Title: {opp.title}
Job Description: {opp.description}

Candidate Profile:
- Name: {user.name}
- Skills: {', '.join(user.skills)}
- Location: {user.city}, {user.province}

Write a concise, professional cover letter in 3-4 paragraphs."""
    
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=str(uuid.uuid4()),
            system_message="You are a professional cover letter writer for South African job applications."
        ).with_model("openai", "gpt-4o-mini")
        
        response = await chat.send_message(UserMessage(text=ai_prompt))
        cover_letter = response
    except Exception as e:
        logger.error(f"AI generation failed: {e}")
        cover_letter = f"Dear Hiring Manager,\n\nI am interested in the {opp.title} position. Please review my CV.\n\nBest regards,\n{user.name}"
    
    # Create application
    application = Application(
        user_id=user_id,
        opportunity_id=req.opportunity_id,
        status="Submitted",
        cv_used=req.cv_id,
        cover_letter=cover_letter,
        ai_prompt_used=ai_prompt,
        match_score=match_score,
        requires_user_action=True  # No real API, user must click apply link
    )
    
    await db.applications.insert_one(application.model_dump())
    
    # Deduct credit (unless admin)
    if not is_admin:
        await db.subscriptions.update_one(
            {"user_id": user_id},
            {"$inc": {"credits_remaining": -1}}
        )
    
    return {
        "success": True,
        "application": application.model_dump(),
        "cover_letter": cover_letter,
        "apply_url": opp.apply_url,
        "message": "Cover letter generated. Click the apply link to submit your application."
    }

@api_router.get("/applications", response_model=List[Dict[str, Any]])
async def get_applications(user_id: str = Depends(get_current_user)):
    # Get user applications
    apps = await db.applications.find({"user_id": user_id}, {"_id": 0}).to_list(1000)
    
    # Enrich with opportunity data
    results = []
    for app_doc in apps:
        app = Application(**app_doc)
        opp_doc = await db.opportunities.find_one({"id": app.opportunity_id}, {"_id": 0})
        if opp_doc:
            opp = Opportunity(**opp_doc)
            results.append({
                **app.model_dump(),
                "opportunity": opp.model_dump()
            })
    
    # Sort by created_at descending
    results.sort(key=lambda x: x["created_at"], reverse=True)
    
    return results

# ============ SUBSCRIPTION ROUTES ============

@api_router.get("/subscription")
async def get_subscription(user_id: str = Depends(get_current_user)):
    sub_doc = await db.subscriptions.find_one({"user_id": user_id}, {"_id": 0})
    if not sub_doc:
        raise HTTPException(status_code=404, detail="Subscription not found")
    return sub_doc

@api_router.post("/subscription/upgrade")
async def upgrade_subscription(req: UpgradePlanRequest, user_id: str = Depends(get_current_user)):
    plans = {
        "Free": {"credits": 3, "price": 0},
        "Starter": {"credits": 20, "price": 49},
        "Premium": {"credits": 99999, "price": 199},
        "VIP": {"credits": 99999, "price": 499}
    }
    
    if req.plan not in plans:
        raise HTTPException(status_code=400, detail="Invalid plan")
    
    # Mock payment (sandbox mode)
    plan_data = plans[req.plan]
    next_billing = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    
    await db.subscriptions.update_one(
        {"user_id": user_id},
        {"$set": {
            "plan": req.plan,
            "credits_remaining": plan_data["credits"],
            "credits_monthly": plan_data["credits"],
            "next_billing_date": next_billing,
            "status": "active"
        }}
    )
    
    return {"success": True, "plan": req.plan, "message": f"Upgraded to {req.plan} plan (sandbox mode)"}

# ============ WALLET & REFERRALS ============

@api_router.get("/wallet")
async def get_wallet(user_id: str = Depends(get_current_user)):
    wallet_doc = await db.wallets.find_one({"user_id": user_id}, {"_id": 0})
    if not wallet_doc:
        raise HTTPException(status_code=404, detail="Wallet not found")
    return wallet_doc

@api_router.get("/referrals/code")
async def get_referral_code(user_id: str = Depends(get_current_user)):
    # Check if user has a referral code
    ref = await db.referrals.find_one({"referrer_user_id": user_id, "referred_user_id": None}, {"_id": 0})
    if ref:
        return {"referral_code": ref["referral_code"]}
    
    # Create new referral code
    ref_code = f"SA{user_id[:8].upper()}"
    referral = Referral(
        referrer_user_id=user_id,
        referral_code=ref_code
    )
    await db.referrals.insert_one(referral.model_dump())
    
    return {"referral_code": ref_code}

@api_router.get("/referrals/stats")
async def get_referral_stats(user_id: str = Depends(get_current_user)):
    refs = await db.referrals.find({"referrer_user_id": user_id}, {"_id": 0}).to_list(1000)
    completed = [r for r in refs if r.get("status") == "completed"]
    
    return {
        "total_referrals": len(refs),
        "completed_referrals": len(completed),
        "pending_referrals": len(refs) - len(completed),
        "total_earned": len(completed) * 50
    }

# ============ ADMIN ROUTES ============

@api_router.get("/admin/users")
async def admin_get_users(user_id: str = Depends(get_current_user)):
    # Check admin
    user_doc = await db.users.find_one({"id": user_id}, {"_id": 0})
    if not user_doc or not user_doc.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    users = await db.users.find({}, {"_id": 0, "password": 0}).to_list(1000)
    return users

@api_router.post("/admin/grant-credits")
async def admin_grant_credits(user_id: str = Depends(get_current_user), target_user_id: str = "", credits: int = 99999):
    # Check admin
    user_doc = await db.users.find_one({"id": user_id}, {"_id": 0})
    if not user_doc or not user_doc.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Grant unlimited credits to self or target
    target_id = target_user_id if target_user_id else user_id
    
    await db.subscriptions.update_one(
        {"user_id": target_id},
        {"$set": {
            "credits_remaining": credits,
            "credits_monthly": credits,
            "plan": "Admin Unlimited"
        }}
    )
    
    return {"success": True, "message": f"Granted {credits} credits"}

@api_router.get("/admin/stats")
async def admin_stats(user_id: str = Depends(get_current_user)):
    # Check admin
    user_doc = await db.users.find_one({"id": user_id}, {"_id": 0})
    if not user_doc or not user_doc.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    total_users = await db.users.count_documents({})
    total_applications = await db.applications.count_documents({})
    total_opportunities = await db.opportunities.count_documents({})
    
    # Subscription breakdown
    subs = await db.subscriptions.find({}, {"_id": 0}).to_list(10000)
    plan_counts = {}
    for sub in subs:
        plan = sub.get("plan", "Free")
        plan_counts[plan] = plan_counts.get(plan, 0) + 1
    
    return {
        "total_users": total_users,
        "total_applications": total_applications,
        "total_opportunities": total_opportunities,
        "plan_distribution": plan_counts
    }

# ============ SEED DATA (for testing) ============

@api_router.post("/seed-opportunities")
async def seed_opportunities():
    """Seed South African job opportunities"""
    existing = await db.opportunities.count_documents({})
    if existing > 0:
        return {"message": "Opportunities already seeded", "count": existing}
    
    sa_jobs = [
        {
            "title": "Software Developer",
            "description": "Full-stack developer needed for fintech startup in Cape Town. React, Node.js, MongoDB experience required.",
            "salary_min": 35000,
            "salary_max": 55000,
            "category": "Technology",
            "source": "CareerJunction",
            "apply_url": "https://www.careerjunction.co.za/apply/software-dev",
            "posted_at": datetime.now(timezone.utc).isoformat(),
            "location": "Cape Town, Western Cape",
            "remote_flag": False,
            "skills_required": ["React", "Node.js", "MongoDB", "JavaScript"]
        },
        {
            "title": "Digital Marketing Manager",
            "description": "Lead digital marketing campaigns for e-commerce company. SEO, SEM, social media expertise required.",
            "salary_min": 25000,
            "salary_max": 40000,
            "category": "Marketing",
            "source": "Indeed",
            "apply_url": "https://za.indeed.com/apply/marketing-manager",
            "posted_at": datetime.now(timezone.utc).isoformat(),
            "location": "Johannesburg, Gauteng",
            "remote_flag": False,
            "skills_required": ["SEO", "SEM", "Google Ads", "Social Media"]
        },
        {
            "title": "Remote Customer Support Agent",
            "description": "Provide customer support via phone and email. Flexible remote work. Fluent in English and Afrikaans preferred.",
            "salary_min": 15000,
            "salary_max": 22000,
            "category": "Customer Service",
            "source": "Gumtree",
            "apply_url": "https://www.gumtree.co.za/apply/customer-support",
            "posted_at": datetime.now(timezone.utc).isoformat(),
            "location": "Remote",
            "remote_flag": True,
            "skills_required": ["Customer Service", "Communication", "Problem Solving"]
        },
        {
            "title": "Accountant",
            "description": "Manage financial records, prepare statements, and ensure compliance. CPA or similar certification required.",
            "salary_min": 28000,
            "salary_max": 45000,
            "category": "Finance",
            "source": "LinkedIn Jobs",
            "apply_url": "https://www.linkedin.com/jobs/apply/accountant-sa",
            "posted_at": datetime.now(timezone.utc).isoformat(),
            "location": "Durban, KwaZulu-Natal",
            "remote_flag": False,
            "skills_required": ["Accounting", "Financial Reporting", "Excel", "Compliance"]
        },
        {
            "title": "Graphic Designer",
            "description": "Create visual content for marketing campaigns. Proficiency in Adobe Creative Suite required.",
            "salary_min": 18000,
            "salary_max": 30000,
            "category": "Design",
            "source": "Upwork",
            "apply_url": "https://www.upwork.com/apply/graphic-designer-sa",
            "posted_at": datetime.now(timezone.utc).isoformat(),
            "location": "Remote",
            "remote_flag": True,
            "skills_required": ["Adobe Photoshop", "Illustrator", "InDesign", "Graphic Design"]
        },
        {
            "title": "Sales Representative",
            "description": "Drive sales for B2B software solutions. Experience in tech sales preferred.",
            "salary_min": 20000,
            "salary_max": 35000,
            "category": "Sales",
            "source": "CareerJunction",
            "apply_url": "https://www.careerjunction.co.za/apply/sales-rep",
            "posted_at": datetime.now(timezone.utc).isoformat(),
            "location": "Pretoria, Gauteng",
            "remote_flag": False,
            "skills_required": ["Sales", "B2B", "Communication", "CRM"]
        },
        {
            "title": "Data Analyst",
            "description": "Analyze business data and provide insights. Python, SQL, and Tableau experience required.",
            "salary_min": 30000,
            "salary_max": 48000,
            "category": "Data",
            "source": "Indeed",
            "apply_url": "https://za.indeed.com/apply/data-analyst",
            "posted_at": datetime.now(timezone.utc).isoformat(),
            "location": "Cape Town, Western Cape",
            "remote_flag": False,
            "skills_required": ["Python", "SQL", "Tableau", "Data Analysis"]
        },
        {
            "title": "Content Writer",
            "description": "Write engaging blog posts and articles for various clients. Remote freelance position.",
            "salary_min": 12000,
            "salary_max": 20000,
            "category": "Writing",
            "source": "Fiverr",
            "apply_url": "https://www.fiverr.com/apply/content-writer",
            "posted_at": datetime.now(timezone.utc).isoformat(),
            "location": "Remote",
            "remote_flag": True,
            "skills_required": ["Content Writing", "SEO", "Research", "Copywriting"]
        },
        {
            "title": "Project Manager",
            "description": "Oversee construction projects from planning to completion. PMP certification preferred.",
            "salary_min": 40000,
            "salary_max": 65000,
            "category": "Project Management",
            "source": "LinkedIn Jobs",
            "apply_url": "https://www.linkedin.com/jobs/apply/project-manager",
            "posted_at": datetime.now(timezone.utc).isoformat(),
            "location": "Johannesburg, Gauteng",
            "remote_flag": False,
            "skills_required": ["Project Management", "PMP", "Construction", "Leadership"]
        },
        {
            "title": "HR Coordinator",
            "description": "Support HR functions including recruitment, onboarding, and employee relations.",
            "salary_min": 22000,
            "salary_max": 32000,
            "category": "Human Resources",
            "source": "Gumtree",
            "apply_url": "https://www.gumtree.co.za/apply/hr-coordinator",
            "posted_at": datetime.now(timezone.utc).isoformat(),
            "location": "Durban, KwaZulu-Natal",
            "remote_flag": False,
            "skills_required": ["HR", "Recruitment", "Communication", "Administration"]
        }
    ]
    
    opportunities = [Opportunity(**job) for job in sa_jobs]
    await db.opportunities.insert_many([opp.model_dump() for opp in opportunities])
    
    return {"message": "Seeded 10 SA job opportunities", "count": len(opportunities)}

# Root route
@api_router.get("/")
async def root():
    return {"message": "AI Money Agent — SA Edition API", "status": "active"}

# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()