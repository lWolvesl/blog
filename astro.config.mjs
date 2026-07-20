import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';

import sitemap from '@astrojs/sitemap';

// https://astro.build/config
export default defineConfig({
	site: 'https://wolves.top',
	integrations: [mdx(), sitemap()],
	redirects: {
		'/tools/ark': { status: 301, destination: 'https://ark.wolves.top' },
	},
	server: {
		host: '::',
		port: 4321, // 你希望使用的端口
	},
});
