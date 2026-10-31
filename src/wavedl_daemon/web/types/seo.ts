import type { MaybeRefOrGetter } from 'vue';

export interface SeoCollectionItem {
  name: string;
  url: string;
}

export interface UseSeoOptions {
  /** Page title */
  title: MaybeRefOrGetter<string | undefined>;
  /** Page description */
  description: MaybeRefOrGetter<string | undefined>;
  /** Page type for og:type (default: 'article' for docs, 'website' for landing) */
  type?: MaybeRefOrGetter<'website' | 'article'>;
  /** Custom OG image URL (absolute) */
  ogImage?: MaybeRefOrGetter<string | undefined>;
  /** Published date for article schema */
  publishedAt?: MaybeRefOrGetter<string | undefined>;
  /** Modified date for article schema */
  modifiedAt?: MaybeRefOrGetter<string | undefined>;
  /** Collection items for CollectionPage schema (e.g. blog listing) */
  collectionItems?: MaybeRefOrGetter<SeoCollectionItem[] | undefined>;
}
