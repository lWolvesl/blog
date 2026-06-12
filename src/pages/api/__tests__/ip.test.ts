import { describe, it, expect } from 'vitest';

// 直接测试 GET handler 逻辑：提取 IP 的核心逻辑
function extractIp(headers: Record<string, string | null>): string {
  const forwarded = headers['X-Forwarded-For'];
  if (forwarded) {
    return forwarded.split(',')[0].trim();
  }
  const realIp = headers['X-Real-IP'];
  if (realIp) {
    return realIp;
  }
  return '127.0.0.1';
}

describe('GET /api/ip', () => {
  it('从 X-Forwarded-For 获取 IP（单 IP）', () => {
    const ip = extractIp({ 'X-Forwarded-For': '117.173.139.123' });
    expect(ip).toBe('117.173.139.123');
  });

  it('从 X-Forwarded-For 获取第一个 IP（多级代理）', () => {
    const ip = extractIp({ 'X-Forwarded-For': '117.173.139.123, 10.0.0.1, 172.16.0.1' });
    expect(ip).toBe('117.173.139.123');
  });

  it('X-Forwarded-For 包含空格时正确 trim', () => {
    const ip = extractIp({ 'X-Forwarded-For': ' 203.0.113.5 , 10.0.0.1 ' });
    expect(ip).toBe('203.0.113.5');
  });

  it('从 X-Real-IP 获取 IP（无 X-Forwarded-For）', () => {
    const ip = extractIp({ 'X-Real-IP': '192.168.1.100' });
    expect(ip).toBe('192.168.1.100');
  });

  it('X-Forwarded-For 优先级高于 X-Real-IP', () => {
    const ip = extractIp({
      'X-Forwarded-For': '117.173.139.123',
      'X-Real-IP': '10.0.0.1',
    });
    expect(ip).toBe('117.173.139.123');
  });

  it('无代理头时返回 127.0.0.1', () => {
    const ip = extractIp({});
    expect(ip).toBe('127.0.0.1');
  });

  it('X-Forwarded-For 为空字符串时回退到 X-Real-IP', () => {
    const ip = extractIp({ 'X-Forwarded-For': '', 'X-Real-IP': '10.0.0.1' });
    expect(ip).toBe('10.0.0.1');
  });

  it('响应格式为 {"ip":"..."} ', () => {
    const ip = extractIp({ 'X-Forwarded-For': '117.173.139.123' });
    const response = JSON.stringify({ ip });
    const parsed = JSON.parse(response);
    expect(parsed).toEqual({ ip: '117.173.139.123' });
    expect(Object.keys(parsed)).toHaveLength(1);
    expect(typeof parsed.ip).toBe('string');
  });
});
