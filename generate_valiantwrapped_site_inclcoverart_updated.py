# --- ONLY SHOWING THE FIXED SECTION ---

def generate_author_page(author_label, summary_row, persona_row, expertise_dir, album_covers_src_dir, author_dir, asset_dir, build_report):
    _, _, _, display_name = parse_author_name(author_label)

    pub = cit = top_journal = top_paper = top_paper_cit = ""
    if summary_row is None:
        build_report.append(
            (author_label, "missing_summary_row", author_label))
    else:
        pub = summary_row.get("pub_count_2025_present", "")
        cit = summary_row.get("citation_count_2025_present", "")
        top_journal = summary_row.get("top_journal_2025_present", "")
        top_paper = summary_row.get("top_paper_title_2025_present", "")
        top_paper_cit = summary_row.get("top_paper_citations_2025_present", "")

    summary_html = f'''
    <div class="grid">
      <div class="card"><div class="label">Publications (2025–Present)</div><div class="value">{html_lib.escape(str(pub))}</div></div>
      <div class="card"><div class="label">Citations (2025–Present)</div><div class="value">{html_lib.escape(str(cit))}</div></div>
      <div class="card"><div class="label">Top Journal</div><div class="small"><b>{html_lib.escape(str(top_journal or ""))}</b></div></div>
      <div class="card"><div class="label">Top Paper</div><div class="small"><b>{html_lib.escape(str(top_paper or ""))}</b><br><span class="pill">Citations: {html_lib.escape(str(top_paper_cit or ""))}</span></div></div>
    </div>
    '''

    expertise_text = read_expertise_text(
        expertise_dir, author_label, build_report)
    expertise_html = html_lib.escape(expertise_text).replace(
        "\n", "<br>") if expertise_text else "No expertise summary generated."

    if persona_row is None:
        build_report.append(
            (author_label, "missing_persona_row", author_label))
        persona_html = "<p class='footer-note'>No persona generated.</p>"
    else:
        artist_name_raw = str(persona_row.get("artist_name", "") or "")
        album_title_raw = str(persona_row.get("album_title", "") or "")
        persona_bio_raw = str(persona_row.get("persona_bio", "") or "")

        # ✅ FIX: compute BEFORE f-string
        persona_bio_html = html_lib.escape(
            persona_bio_raw).replace("\n", "<br>")

        ok, rel_path = copy_album_cover_into_docs(
            album_covers_src_dir, asset_dir, author_label, build_report)

        cover_block = (
            f'<div class="cover-wrap"><img class="album-cover" src="{rel_path}" alt="{html_lib.escape((artist_name_raw + " " + album_title_raw).strip())}" loading="lazy"></div>'
            if ok else
            '<div class="cover-placeholder">Album cover art not available yet.</div>'
        )

        persona_html = f'''
        <div class="album-card">
          <div class="artist">{html_lib.escape(artist_name_raw)}</div>
          <p class="bio">{persona_bio_html}</p>
          <div class="album-title">Album: <span>{html_lib.escape(album_title_raw)}</span></div>
          {cover_block}
          {format_tracklist(persona_row.get("tracklist", ""))}
        </div>
        '''
