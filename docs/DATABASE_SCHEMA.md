# 1. DATABASE SCHEMA

## 1.1 Table: **user**

| Field                  | Type          | Default           | Description                      |
|------------------------|---------------|-------------------|----------------------------------|
| `id`                   | bigint        | -                 | **Primary Key** - User ID        |
| `created_date`         | datetime      | CURRENT_TIMESTAMP | Created date                     |
| `last_modified_date`   | datetime      | CURRENT_TIMESTAMP | Last time update any information |
| `date_of_birth`        | date          | null              | Date of birth                    |
| `email`                | varchar(255)  | ''                | Email (encrypted)                |
| `gender`               | int           | 0                 | Gender                           |
| `profile_picture_url`  | varchar(2048) | ''                | Profile picture URL              |
| `last_login`           | datetime      | CURRENT_TIMESTAMP | Last login                       |
| `phone_number`         | bigint        | 0                 | Phone number (encrypted)         |
| `status`               | int           | 0                 | Status (Active/Inactive/Deleted) |
| `username`             | varchar(255)  | ''                | Username                         |
| `first_name`           | varchar(255)  | ''                | First name                       |
| `last_name`            | varchar(255)  | ''                | Last name                        |
| `profile_type`         | tinyint(1)    | 0                 | Profile type (User 0/Admin 1)    |

Indexes:
User_email_index (email)
User_phone_number_index (phone_number)

## 1.2 Table: **user_password**

| Field                  | Type         | Default | Description                               |
|------------------------|--------------|---------|-------------------------------------------|
| `user_id`              | bigint       | -       | **Primary Key** - User ID                 |
| `identifier`           | varchar(255) | ''      | **Primary Key** - Email/Phone (encrypted) |
| `type`                 | tinyint(1)   | 0       | **Primary Key** - Type (Email 0/Phone 1)  |
| `password_algo`        | varchar(255) | ''      | The algorithm used for hashing (bcrypt)   |
| `password`             | varchar(255) | ''      | Password (hashed)                         |
| `password_salt`        | varchar(255) | ''      | Salt                                      |
| `password_reset_token` | varchar(255) | ''      | Token used to reset password              |

## 1.3 Table: **otp**

| Field               | Type         | Default | Description                           |
|---------------------|--------------|---------|---------------------------------------|
| `id`                | bigint       | -       | **Primary Key** - OTP ID              |
| `receiver`          | varchar(255) | ''      | Email/Phone to receive OTP            |
| `receiver_type`     | tinyint(1)   | 0       | Receiver type (Email 0/Phone 1)       |
| `user_id`           | bigint       | 0       | User ID                               |
| `otp_expired_at`    | bigint       | 0       | OTP expiration time                   |
| `otp_attempt`       | tinyint(1)   | 0       | Number of OTP retry attempts          |
| `resend_attempt`    | tinyint(1)   | 0       | Number of OTP resend attempts         |
| `resend_expired_at` | bigint       | 0       | Resend expiration time                |
| `activation_code`   | varchar(16)  | ''      | Activation code                       |
| `state`             | tinyint      | 0       | State (Active 0/Inactive 1/Blocked 2) |
| `created_at`        | bigint       | 0       | Activation code creation timestamp    |
| `updated_at`        | bigint       | 0       | Activation code update timestamp      |

## 1.4 Table: **training_configs**

| Field            | Type         | Default           | Description                                    |
|------------------|--------------|-------------------|------------------------------------------------|
| `id`             | serial       | -                 | **Primary Key** - Config ID                    |
| `config_name`    | varchar(255) | -                 | Name of the training configuration             |
| `config_version` | varchar(50)  | '1.0'             | Version of the configuration                   |
| `created_by`     | varchar(100) | null              | User who created the configuration             |
| `created_at`     | timestamp    | CURRENT_TIMESTAMP | Configuration creation timestamp               |
| `updated_at`     | timestamp    | CURRENT_TIMESTAMP | Configuration last update timestamp            |
| `description`    | text         | null              | Description of the configuration               |
| `tags`           | text[]       | null              | Tags for categorization                        |
| `is_default`     | boolean      | false             | Flag indicating if this is the default config  |
| `config`         | jsonb        | -                 | Training configuration in JSONB format         |
| `config_hash`    | varchar(64)  | (generated)       | SHA-256 hash of config (auto-generated)        |

Constraints:
unique_config_hash (config_hash)
