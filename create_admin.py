#!/usr/bin/env python3
"""
Script to create an admin user
Usage: python create_admin.py
"""
import sys
sys.path.insert(0, '.')

from database.db import SessionLocal
from database.crud import create_user
from sqlalchemy.exc import IntegrityError

def create_admin():
    db = SessionLocal()
    
    email = input("Enter admin email (default: admin@fraud.com): ").strip() or "admin@fraud.com"
    password = input("Enter admin password (default: admin123): ").strip() or "admin123"
    
    try:
        user = create_user(db, email, password, role="admin")
        print(f"\n✅ Admin user created successfully!")
        print(f"   Email: {user.email}")
        print(f"   Role: {user.role}")
        print(f"   ID: {user.id}")
        print(f"\nYou can now login with these credentials on the dashboard.")
    except IntegrityError:
        print(f"\n❌ Error: User with email {email} already exists!")
        print("Please use a different email or delete the existing user first.")
    except Exception as e:
        print(f"\n❌ Error creating admin user: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    create_admin()
