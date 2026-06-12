import type { APIRoute } from 'astro';

export const GET: APIRoute = ({ request }) => {
  // 从 X-Forwarded-For 或直接连接获取客户端真实 IP
  const forwarded = request.headers.get('X-Forwarded-For');
  const ip = forwarded?.split(',')[0]?.trim()
    || request.headers.get('X-Real-IP')
    || '127.0.0.1';

  return new Response(JSON.stringify({ ip }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
};
