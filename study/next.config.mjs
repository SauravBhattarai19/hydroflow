import createMDX from '@next/mdx';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';

const repoBasePath = process.env.GITHUB_PAGES === 'true' ? '/hydroflow' : '';

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',           // static export → GitHub Pages
  pageExtensions: ['js', 'jsx', 'ts', 'tsx', 'md', 'mdx'],
  images: { unoptimized: true },
  basePath: repoBasePath,
  assetPrefix: repoBasePath,
};

const withMDX = createMDX({
  options: {
    remarkPlugins: [remarkMath, remarkGfm],
    rehypePlugins: [rehypeKatex],
  },
});

export default withMDX(nextConfig);
