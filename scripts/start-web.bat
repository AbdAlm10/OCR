@echo off
cd /d "%~dp0..\web"
if not exist .env.local copy .env.local.example .env.local
npm run dev
