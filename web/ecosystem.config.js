module.exports = {
  apps: [
    {
      name: 'nim-chat',
      cwd: '/datos/repos/deebseckv4/web',
      script: 'python3',
      args: '-m uvicorn app:app --host 0.0.0.0 --port 47821',
      interpreter: 'none',
      autorestart: true,
      max_restarts: 10,
      restart_delay: 3000,
      env: {
        PYTHONUNBUFFERED: '1',
      },
      out_file: '/home/stev/.pm2/logs/nim-chat-out.log',
      error_file: '/home/stev/.pm2/logs/nim-chat-error.log',
      merge_logs: true,
      time: true,
    },
  ],
};
