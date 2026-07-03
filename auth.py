import json

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from db import db
from models import Analysis, User


auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        if not email or not password or not confirm_password:
            flash("Email, password, and confirm password are required")
            return render_template("register.html"), 400

        if password != confirm_password:
            flash("Passwords do not match")
            return render_template("register.html"), 400

        existing_user = User.query.filter_by(email=email).first()
        if existing_user is not None:
            flash("Email is already taken")
            return render_template("register.html"), 400

        user = User(email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for("auth.login"))

    return render_template("register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        user = User.query.filter_by(email=email).first()
        if user is not None and user.check_password(password):
            login_user(user)
            return redirect(url_for("auth.dashboard"))

        flash("Invalid credentials")
        return render_template("login.html"), 401

    return render_template("login.html")


@auth_bp.route("/logout", methods=["GET"])
def logout():
    logout_user()
    return redirect(url_for("auth.login"))


@auth_bp.route("/dashboard", methods=["GET"])
@login_required
def dashboard():
    analyses = (
        Analysis.query.filter_by(user_id=current_user.id)
        .order_by(Analysis.created_at.desc())
        .all()
    )

    analysis_rows = []
    for analysis in analyses:
        results = json.loads(analysis.results_json) if analysis.results_json else {}
        analysis_rows.append(
            {
                "date": analysis.created_at,
                "assets": analysis.assets,
                "capital": analysis.capital,
                "sharpe_ratio": results.get("sharpe_ratio"),
                "win_probability": results.get("win_probability"),
            }
        )

    return render_template("dashboard.html", analyses=analysis_rows)