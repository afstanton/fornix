//! Pagination types for list/scan operations across all storage backends.

/// Parameters for a paginated list request.
#[derive(Debug, Clone)]
pub struct PageParams {
    /// Maximum number of records to return.
    pub limit: usize,
    /// Opaque cursor returned by a previous page response.
    /// `None` means start from the beginning.
    pub cursor: Option<String>,
}

impl PageParams {
    /// Start from the beginning with the given limit.
    pub fn first(limit: usize) -> Self {
        Self { limit, cursor: None }
    }

    /// Continue from a cursor returned by a previous page.
    pub fn after(cursor: impl Into<String>, limit: usize) -> Self {
        Self {
            limit,
            cursor: Some(cursor.into()),
        }
    }
}

impl Default for PageParams {
    fn default() -> Self {
        Self::first(100)
    }
}

/// A single page of results from a list/scan operation.
#[derive(Debug, Clone)]
pub struct Page<T> {
    /// The records in this page.
    pub items: Vec<T>,
    /// Cursor to pass to the next request, if more records exist.
    /// `None` means this is the last page.
    pub next_cursor: Option<String>,
    /// Total number of matching records, if the backend can provide it cheaply.
    /// `None` means the total is not known without a separate count query.
    pub total: Option<usize>,
}

impl<T> Page<T> {
    /// Construct a page with a known next cursor.
    pub fn with_cursor(items: Vec<T>, next_cursor: impl Into<String>, total: Option<usize>) -> Self {
        Self {
            items,
            next_cursor: Some(next_cursor.into()),
            total,
        }
    }

    /// Construct a final page with no further results.
    pub fn last(items: Vec<T>, total: Option<usize>) -> Self {
        Self {
            items,
            next_cursor: None,
            total,
        }
    }

    /// Returns `true` if there are more pages after this one.
    pub fn has_next(&self) -> bool {
        self.next_cursor.is_some()
    }

    /// Map the items in this page to a different type.
    pub fn map<U, F: Fn(T) -> U>(self, f: F) -> Page<U> {
        Page {
            items: self.items.into_iter().map(f).collect(),
            next_cursor: self.next_cursor,
            total: self.total,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- PageParams ---

    #[test]
    fn first_has_no_cursor() {
        let p = PageParams::first(50);
        assert_eq!(p.limit, 50);
        assert!(p.cursor.is_none());
    }

    #[test]
    fn default_is_first_100() {
        let p = PageParams::default();
        assert_eq!(p.limit, 100);
        assert!(p.cursor.is_none());
    }

    #[test]
    fn after_stores_cursor_and_limit() {
        let p = PageParams::after("cursor-abc", 25);
        assert_eq!(p.limit, 25);
        assert_eq!(p.cursor.as_deref(), Some("cursor-abc"));
    }

    #[test]
    fn after_accepts_owned_string_cursor() {
        let p = PageParams::after(String::from("xyz"), 10);
        assert_eq!(p.cursor.as_deref(), Some("xyz"));
    }

    // --- Page ---

    #[test]
    fn with_cursor_sets_next_cursor() {
        let page: Page<i32> = Page::with_cursor(vec![1, 2, 3], "next-cursor", Some(10));
        assert_eq!(page.items, vec![1, 2, 3]);
        assert_eq!(page.next_cursor.as_deref(), Some("next-cursor"));
        assert_eq!(page.total, Some(10));
        assert!(page.has_next());
    }

    #[test]
    fn last_has_no_next_cursor() {
        let page: Page<&str> = Page::last(vec!["a", "b"], Some(2));
        assert_eq!(page.items, vec!["a", "b"]);
        assert!(page.next_cursor.is_none());
        assert_eq!(page.total, Some(2));
        assert!(!page.has_next());
    }

    #[test]
    fn last_without_total() {
        let page: Page<u8> = Page::last(vec![], None);
        assert!(page.total.is_none());
        assert!(!page.has_next());
    }

    #[test]
    fn empty_page_has_no_next() {
        let page: Page<i32> = Page::last(vec![], None);
        assert!(!page.has_next());
        assert!(page.items.is_empty());
    }

    #[test]
    fn map_transforms_items_preserving_cursor_and_total() {
        let page: Page<i32> = Page::with_cursor(vec![1, 2, 3], "cur", Some(10));
        let mapped = page.map(|n| n.to_string());
        assert_eq!(mapped.items, vec!["1", "2", "3"]);
        assert_eq!(mapped.next_cursor.as_deref(), Some("cur"));
        assert_eq!(mapped.total, Some(10));
    }

    #[test]
    fn map_on_last_page_preserves_no_cursor() {
        let page: Page<i32> = Page::last(vec![10, 20], None);
        let mapped = page.map(|n| n * 2);
        assert_eq!(mapped.items, vec![20, 40]);
        assert!(mapped.next_cursor.is_none());
    }

    #[test]
    fn page_clone_is_independent() {
        let page: Page<i32> = Page::with_cursor(vec![1], "c", None);
        let cloned = page.clone();
        assert_eq!(cloned.items, page.items);
        assert_eq!(cloned.next_cursor, page.next_cursor);
    }
}
