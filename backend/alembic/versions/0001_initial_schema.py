"""initial schema: users, refresh_tokens, images, model_registry, assessment_results

Revision ID: 0001
Revises:
Create Date: 2026-11-12
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

JSONB = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
_HEADS = ("technical_quality", "anatomical_completeness", "artifact", "orientation", "diagnostic_suitability")


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("username", sa.String(50), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role", sa.String(20), server_default="radiologist", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("failed_login_attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("role IN ('radiologist', 'researcher', 'admin')", name="ck_users_role"),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"])

    op.create_table(
        "images",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("original_filename", sa.String(255), nullable=False),
        sa.Column("file_path", sa.String(500), nullable=False),
        sa.Column("thumbnail_path", sa.String(500), nullable=False),
        sa.Column("preview_path", sa.String(500), nullable=False),
        sa.Column("file_format", sa.String(10), nullable=False),
        sa.Column("modality", sa.String(20), nullable=True),
        sa.Column("modality_source", sa.String(10), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("upload_timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("file_format IN ('DICOM', 'PNG', 'JPEG')", name="ck_images_file_format"),
        sa.CheckConstraint("modality IS NULL OR modality IN ('xray', 'ct', 'mri', 'retinal')", name="ck_images_modality"),
        sa.CheckConstraint(
            "modality_source IS NULL OR modality_source IN ('user', 'dicom', 'model')", name="ck_images_modality_source"
        ),
    )
    op.create_index("ix_images_user_uploaded", "images", ["user_id", "upload_timestamp"])

    op.create_table(
        "model_registry",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("version", sa.String(50), nullable=False, unique=True),
        sa.Column("checkpoint_path", sa.String(500), nullable=False),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("metrics", JSONB, nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("deployed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(
        "uq_model_registry_single_active", "model_registry", ["is_active"], unique=True,
        postgresql_where=sa.text("is_active"), sqlite_where=sa.text("is_active"),
    )

    op.create_table(
        "assessment_results",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("image_id", sa.Uuid(), sa.ForeignKey("images.id", ondelete="CASCADE"), nullable=False),
        sa.Column("model_id", sa.Uuid(), sa.ForeignKey("model_registry.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("status", sa.String(12), nullable=False),
        sa.Column("error_code", sa.String(50), nullable=True),
        sa.Column("error_message", sa.String(500), nullable=True),
        *[sa.Column(f"{h}_score", sa.Float(), nullable=True) for h in _HEADS],
        *[sa.Column(f"{h}_confidence", sa.Float(), nullable=True) for h in _HEADS],
        sa.Column("ai_readiness_score", sa.Float(), nullable=True),
        sa.Column("verdict", sa.String(10), nullable=True),
        sa.Column("artifact_present", sa.Boolean(), nullable=True),
        sa.Column("out_of_distribution", sa.Boolean(), nullable=True),
        sa.Column("details", JSONB, nullable=True),
        sa.Column("inference_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('processing', 'completed', 'error')", name="ck_assessment_status"),
        sa.CheckConstraint("verdict IS NULL OR verdict IN ('Pass', 'Review', 'Reject')", name="ck_assessment_verdict"),
    )
    op.create_index("ix_assessment_image_created", "assessment_results", ["image_id", "created_at"])
    op.create_index("ix_assessment_results_model_id", "assessment_results", ["model_id"])



def downgrade() -> None:
    op.drop_index("ix_assessment_results_model_id", table_name="assessment_results")
    op.drop_index("ix_assessment_image_created", table_name="assessment_results")
    op.drop_table("assessment_results")
    op.drop_index("uq_model_registry_single_active", table_name="model_registry")
    op.drop_table("model_registry")
    op.drop_index("ix_images_user_uploaded", table_name="images")
    op.drop_table("images")
    op.drop_index("ix_refresh_tokens_user_id", table_name="refresh_tokens")
    op.drop_table("refresh_tokens")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")
