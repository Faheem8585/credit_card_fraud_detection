#!/usr/bin/env python3
"""
Initialize default admin user for the fraud detection system.
Run this once after setting up the database.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import SessionLocal, init_db
from database import crud

def create_admin_user():
    """Create default admin user if it doesn't exist"""
    init_db()  # Ensure tables exist
    
    db = SessionLocal()
    try:
        # List of admin accounts to create
        admin_accounts = [
            {"email": "admin@fraud-detection.com", "password": "admin123"},
            {"email": "admin2@fraud.com", "password": "admin123"}
        ]
        
        for admin_info in admin_accounts:
            # Check if admin exists
            existing_admin = crud.get_user_by_email(db, admin_info["email"])
            
            if existing_admin:
                print(f"✅ Admin user already exists: {admin_info['email']}")
                # Update role to admin if not already
                if existing_admin.role != "admin":
                    existing_admin.role = "admin"
                    db.commit()
                    print(f"   ✅ Updated role to admin")
            else:
                # Create new admin user
                admin_user = crud.create_user(db, admin_info["email"], admin_info["password"], role="admin")
                print(f"✅ Created admin user: {admin_info['email']}")
                print(f"   Password: {admin_info['password']}")
        
        print("\n🎉 Admin setup complete!")
        print("\nAdmin Credentials:")
        for admin_info in admin_accounts:
            print(f"   Email: {admin_info['email']}")
            print(f"   Password: {admin_info['password']}")
        print("\n⚠️  Please change the passwords after first login!")
        
    except Exception as e:
        print(f"❌ Error creating admin user: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    create_admin_user()
