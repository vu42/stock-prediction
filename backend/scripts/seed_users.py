"""
Script to seed the database with test users.
Run with: docker exec stock-prediction-api python -m scripts.seed_users
"""

from sqlalchemy import select

from app.core.security import get_password_hash
from app.db.models import User, UserRole
from app.db.session import SessionLocal

# Test users as specified in SPECS.md
TEST_USERS = [
    {
        "username": "enduser1",
        "password": "pass1234",
        "display_name": "End User 1",
        "role": UserRole.END_USER.value,
        "email": "enduser1@example.com",
    },
    {
        "username": "enduser2",
        "password": "pass1234",
        "display_name": "End User 2",
        "role": UserRole.END_USER.value,
        "email": "enduser2@example.com",
    },
    {
        "username": "ds1",
        "password": "pass1234",
        "display_name": "Data Scientist 1",
        "role": UserRole.DATA_SCIENTIST.value,
        "email": "ds1@example.com",
    },
    {
        "username": "ds2",
        "password": "pass1234",
        "display_name": "Data Scientist 2",
        "role": UserRole.DATA_SCIENTIST.value,
        "email": "ds2@example.com",
    },
    {
        "username": "admin",
        "password": "admin123",
        "display_name": "Admin User",
        "role": UserRole.ADMIN.value,  # Admin can access DS features
        "email": "admin@example.com",
    },
]


def seed_users():
    """Seed the database with test users."""
    db = SessionLocal()
    try:
        created_count = 0
        updated_count = 0
        skipped_count = 0
        
        for user_data in TEST_USERS:
            username = user_data["username"]
            
            # Check if user already exists
            stmt = select(User).where(User.username == username)
            existing = db.execute(stmt).scalar_one_or_none()
            
            if existing:
                # Update existing user (update password hash in case it changed)
                existing.password_hash = get_password_hash(user_data["password"])
                existing.display_name = user_data["display_name"]
                existing.role = user_data["role"]
                existing.email = user_data["email"]
                existing.is_active = True
                updated_count += 1
                print(f"✓ Updated: {username} ({user_data['role']})")
            else:
                # Create new user
                user = User(
                    username=username,
                    password_hash=get_password_hash(user_data["password"]),
                    display_name=user_data["display_name"],
                    role=user_data["role"],
                    email=user_data["email"],
                    is_active=True,
                )
                db.add(user)
                created_count += 1
                print(f"✓ Created: {username} ({user_data['role']}) - Password: {user_data['password']}")
        
        db.commit()
        
        print(f"\n✅ Seeding complete!")
        print(f"   Created: {created_count} users")
        print(f"   Updated: {updated_count} users")
        print(f"   Total: {len(TEST_USERS)} users")
        print(f"\n📋 Test Accounts:")
        print(f"   End Users:")
        print(f"     - enduser1 / pass1234")
        print(f"     - enduser2 / pass1234")
        print(f"   Data Scientists:")
        print(f"     - ds1 / pass1234")
        print(f"     - ds2 / pass1234")
        print(f"     - admin / admin123")
        
    except Exception as e:
        print(f"❌ Error seeding users: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_users()

