import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';

import sitemap from '@astrojs/sitemap';

// https://astro.build/config
export default defineConfig({
	site: 'https://wolves.top',
	integrations: [mdx(), sitemap()],
	server: {
		host: '::',
		port: 4321, // 你希望使用的端口
	},
});
